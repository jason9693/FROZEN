import pytorch_lightning as pl
import torch
from einops import repeat
from einops.layers.torch import Rearrange
from timm.utils import AverageMeter
from torch import nn
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer
import timm

from frozen.objectives import SacreBLEU
from utils import freeze_module


class ModalityTranslator(pl.LightningModule):
    def __init__(
        self,
        tokenizer=None,
        vision_encoder_path='nf_resnet50',
        use_pretrained_vision_encoder=True,
        freeze_vision_encoder=True,
        num_frozen_stages=3,
        opt_type='adam',
        sched_type=None,
        sched_milestones=None,
        learning_rate=1e-4,
        bleu_conf=None
    ):
        super().__init__()
        self.vit_path = vision_encoder_path
        self.encoder = timm.create_model(vision_encoder_path, pretrained=use_pretrained_vision_encoder)
        m2_model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
        self.decoder = m2_model.model.decoder
        if tokenizer is None:
            self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
        else:
            self.tokenizer = tokenizer
        self.bleu_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.lm_config = self.decoder.config
        self.embed_dim = self.lm_config.d_model
        out_dim = self.encoder.num_features
        self.encoder_head = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, self.embed_dim, 3, 2, 1),
            Rearrange('b d h w -> b (h w) d')
        )
        self.decoder_head = m2_model.lm_head
        if freeze_vision_encoder:
            freeze_module(self.encoder.stem)
            for block in self.encoder.stages[:num_frozen_stages]:
                freeze_module(block)
        freeze_module(self.decoder)
        bos_sep_token_id = torch.tensor([self.tokenizer.bos_token_id, self.tokenizer.sep_token_id]).view(1, -1)
        with torch.no_grad():
            bos_sep_embed = self.decoder.embed_tokens(bos_sep_token_id)*self.decoder.embed_scale
            bos_embed = bos_sep_embed[:, :1]
            sep_embed = bos_sep_embed[:, 1:]
            bos_embed.requires_grad = False
            sep_embed.requires_grad = False
        self.register_buffer('bos_embed', bos_embed)
        self.register_buffer('sep_embed', sep_embed)
        default_bleu_kwargs = dict(n_gram=2)
        if bleu_conf is not None:
            default_bleu_kwargs.update(bleu_conf)
        self.bleu = SacreBLEU(
            model_tokenizer=self.tokenizer,
            metric_tokenizer=self.bleu_tokenizer,
            **default_bleu_kwargs
        )
        self.opt_type = opt_type
        self.learning_rate = learning_rate
        self.sched_type = sched_type
        self.sched_milestones = sched_milestones
        self.meters = {'m2/train_bleu': AverageMeter(), 'm2/val_bleu': AverageMeter()}

    def forward(self, img, input_ids, attention_mask=None, separate=True):
        vision_embeds = self.forward_vision_encoder(img, separate)
        return self.forward_language_decoder(vision_embeds, input_ids, attention_mask)

    def forward_vision_encoder(self, img, separate=True):
        vision_embeds = self.encoder_head(self.encoder.forward_features(img))
        bos_embed = repeat(self.bos_embed, '() n d -> b n d', b=img.size(0))
        vision_embeds = torch.cat([bos_embed, vision_embeds], dim=1)
        if separate:
            sep_embed = repeat(self.sep_embed, '() n d -> b n d', b=img.size(0))
            vision_embeds = torch.cat([vision_embeds, sep_embed], dim=1)
        return vision_embeds

    def forward_language_decoder(self, vision_embeds, input_ids, attention_mask=None):
        kwargs = dict(
            encoder_hidden_states=vision_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.decoder_head(self.decoder(**kwargs).last_hidden_state)

    def compute_loss(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        criterion = nn.CrossEntropyLoss()
        logits = self(img, input_ids[:, :-1], attention_mask[:, :-1])
        loss = criterion(logits.view(-1, self.lm_config.vocab_size), input_ids[:, 1:].flatten())
        return loss

    # compute bleu using teacher-forcing outputs
    @torch.no_grad()
    def compute_fast_bleu(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        logits = self(img, input_ids[:, :-1], attention_mask[:, :-1])
        output_ids = logits.argmax(dim=-1)
        return self.bleu.score(input_ids[:, 1:], output_ids)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        fast_bleu = self.compute_fast_bleu(batch)
        bleu_meter = self.meters['m2/train_bleu']
        bleu_meter.update(fast_bleu)
        self.log('m2/train_bleu', torch.tensor(bleu_meter.avg), prog_bar=True)
        self.log('m2/train_loss', torch.tensor(loss.item()))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        fast_bleu = self.compute_fast_bleu(batch)
        bleu_meter = self.meters['m2/val_bleu']
        bleu_meter.update(fast_bleu)
        self.log('m2/val_bleu', torch.tensor(bleu_meter.avg), prog_bar=True)
        self.log('m2/val_loss', torch.tensor(loss.item()))
        return fast_bleu

    def on_train_epoch_start(self):
        self.meters['m2/train_bleu'].reset()

    def on_validation_epoch_start(self):
        self.meters['m2/val_bleu'].reset()

    def configure_optimizers(self):
        if self.opt_type == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.95))
        elif self.opt_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True)
        else:
            raise KeyError
        if self.sched_type == 'None':
            return optimizer
        elif self.sched_type == 'multistep':
            if self.sched_milestones is None:
                raise AttributeError
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.sched_milestones, gamma=0.1)
        elif self.sched_type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        else:
            raise KeyError
        return [optimizer], [scheduler]

    @torch.no_grad()
    def infer(self, img, max_length, ignore_eos=False):
        decoding_output = torch.tensor([[self.tokenizer.get_lang_id('en')]]).long().cuda()
        for i in range(max_length):
            logits = self(img, decoding_output.view(1, -1))
            decoding_output = logits[0].argmax(dim=0)[-i-1:]
            next_token = decoding_output[-1]
            if next_token == self.tokenizer.eos_token_id and not ignore_eos:
                return decoding_output
        return decoding_output



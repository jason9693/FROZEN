import pytorch_lightning as pl
import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import timm

from utils import freeze_module


class ModalityTranslator(pl.LightningModule):
    def __init__(
        self,
        tokenizer=None,
        vision_encoder_path='nf_resnet50',
        num_frozen_stages=3,
        opt_type='adam'
    ):
        super().__init__()
        self.vit_path = vision_encoder_path
        self.encoder = timm.create_model(vision_encoder_path, pretrained=True)
        m2_model = M2M100ForConditionalGeneration.from_pretrained('facebook/m2m100_418M')
        self.decoder = m2_model.model.decoder
        if tokenizer is None:
            self.tokenizer = M2M100Tokenizer.from_pretrained('facebook/m2m100_418M')
        else:
            self.tokenizer = tokenizer
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
        self.opt_type = opt_type

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

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('m2/train_loss', torch.tensor(loss.item()))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('m2/val_loss', torch.tensor(loss.item()))
        return loss

    def configure_optimizers(self):
        if self.opt_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.9, 0.95))
        elif self.opt_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=2e-3, momentum=0.9, nesterov=True)
        else:
            raise KeyError
        return optimizer

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



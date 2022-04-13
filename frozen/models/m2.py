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
        # TODO: positional information for vision encoder
        self.decoder_head = m2_model.lm_head
        for block in self.encoder.stages[:num_frozen_stages]:
            freeze_module(block)
        freeze_module(self.decoder)
        sep_token_id = torch.tensor(self.tokenizer.sep_token_id).view(1, 1)
        with torch.no_grad():
            sep_embed = self.decoder.embed_tokens(sep_token_id)*self.decoder.embed_scale
            sep_embed.requires_grad = False
        self.register_buffer('sep_embed', sep_embed)
        self.opt_type = opt_type

    def forward(self, img, input_ids, attention_mask, separate=True):
        vision_embeds = self.forward_vision_encoder(img, separate)
        return self.forward_language_decoder(vision_embeds, input_ids, attention_mask)

    def forward_vision_encoder(self, img, separate=True):
        vision_embeds = self.encoder_head(self.encoder.forward_features(img))
        if separate:
            sep_embed = repeat(self.sep_embed, '() n d -> b n d', b=img.size(0))
            vision_embeds = torch.cat([vision_embeds, sep_embed], dim=1)
        return vision_embeds

    def forward_language_decoder(self, vision_embeds, input_ids, attention_mask):
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
        vision_embeds = self.forward_vision_encoder(img)
        criterion = nn.CrossEntropyLoss()
        logits = self.forward_language_decoder(vision_embeds, input_ids[:, :-1], attention_mask[:, :-1])
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
            optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.95))
        elif self.opt_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
        else:
            raise KeyError
        return optimizer



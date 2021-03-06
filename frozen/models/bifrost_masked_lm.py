import torch
from torch import nn
from transformers import AutoModelForMaskedLM

from frozen.models.base import BiFrostBase
from frozen.models.layers import conv3_bn_gelu
from utils import freeze_module


class BiFrostMaskedLMBase(BiFrostBase):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self.encoder_head = nn.Sequential(
            *(conv3_bn_gelu(self.encoder.num_features, self.embed_dim, 2) for _ in range(config.num_ds))
        )

    def _set_decoder(self):
        self.decoder = AutoModelForMaskedLM.from_pretrained(self.config.lm_path)
        self.lm_config = self.decoder.config
        freeze_module(self.decoder)

    def forward(self, img, input_ids, attention_mask=None):
        vision_embeds = self.forward_encoder(img)
        return self.forward_language_model(vision_embeds, input_ids)

    def forward_encoder(self, img):
        vision_embeds = self.encoder_head(self.encoder.forward_features(img))
        return vision_embeds

    def forward_language_model(self, vision_embeds, input_ids):
        lm_embeds = self.decoder.model.get_input_embeddings()(input_ids)
        embeds = torch.cat([vision_embeds, lm_embeds], dim=1)
        kwargs = dict(input_embeds=embeds)
        return self.decoder(**kwargs)

    def compute_loss(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids_mlm']
        labels = batch['text_labels_mlm']
        criterion = nn.CrossEntropyLoss()
        logits = self(img, input_ids)
        vision_pad_size = (logits.shape[0], logits.shape[1]-input_ids.shape[1])
        vision_pad = torch.full(vision_pad_size, self.tokenizer.pad_token_id).to(self.device)
        target = torch.cat([vision_pad, labels], dim=1)
        loss = criterion(logits.permute(0, 2, 1), target)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train/loss', torch.tensor(loss.item()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val/loss', torch.tensor(loss.item()), prog_bar=True)
        return loss

    @torch.no_grad()
    def infer(self, img, input_ids):
        assert img.size(0) == 1, 'The inference for batch is not supported yet.'
        logits = self(img, input_ids)
        output = logits[0].argmax(dim=-1)
        return output


class BiFrostBERT(BiFrostMaskedLMBase):
    def forward_language_model(self, vision_embeds, input_ids):
        return super().forward_language_model(vision_embeds, input_ids).logits


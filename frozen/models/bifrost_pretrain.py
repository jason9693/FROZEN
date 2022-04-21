import torch
from torch import nn
from transformers import AutoModelForPreTraining

from frozen.models.base import BiFrostBase
from utils import freeze_module


class BiFrostForPreTrainingBase(BiFrostBase):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)

    def _set_decoder(self):
        self.decoder = AutoModelForPreTraining.from_pretrained(self.config.lm_path)
        self.lm_config = self.decoder.config
        freeze_module(self.decoder)

    def forward(self, img, input_ids, attention_mask=None):
        vision_embeds = self.forward_vision_encoder(img)
        return self.forward_language_model(vision_embeds, input_ids)

    def forward_vision_encoder(self, img):
        vision_embeds = self.encoder_head(self.encoder.forward_features(img))
        return vision_embeds

    def forward_language_model(self, vision_embeds, input_ids):
        lm_embeds = self.decoder.model.get_input_embeddings()(input_ids)
        # todo: optionally choose vision-first, vision-last
        embeds = torch.cat([lm_embeds[:, :1], vision_embeds, lm_embeds[:, 1:]], dim=1)
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
        # except first token(cls)
        target = torch.cat([vision_pad, labels[:, 1:]], dim=1)
        loss = criterion(logits[:, 1:].permute(0, 2, 1), target)
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
        output = logits[0, 1:].argmax(dim=-1)
        return output


class BiFrostBERTForPreTraining(BiFrostForPreTrainingBase):
    def forward_language_model(self, vision_embeds, input_ids):
        return super().forward_language_model(vision_embeds, input_ids).prediction_logits


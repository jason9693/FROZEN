import torch
from torch import nn
from transformers import AutoModelForCausalLM

from frozen.models.base import BiFrostVisionEncBase
from frozen.models.layers import conv3_bn_gelu
from utils import freeze_module


class BiFrostCausalLMBase(BiFrostVisionEncBase):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self.encoder_head = nn.Sequential(
            *(conv3_bn_gelu(self.encoder.num_features, self.embed_dim, 2) for _ in range(config.num_ds))
        )
        self._set_bleu_metric()

    def _set_decoder(self):
        self.decoder = AutoModelForCausalLM.from_pretrained(self.config.lm_path)
        self.lm_config = self.decoder.config
        freeze_module(self.decoder)

    def forward(self, img, input_ids, attention_mask=None):
        vision_embeds = self.forward_encoder(img)
        return self.forward_language_model(vision_embeds, input_ids, attention_mask)

    def forward_encoder(self, img):
        vision_embeds = self.encoder_head(self.encoder.forward_features(img))
        return vision_embeds

    def forward_language_model(self, vision_embeds, input_ids, attention_mask=None):
        lm_embeds = self.decoder.model.get_input_embeddings()(input_ids)
        # todo: optionally choose vision-first, vision-last
        embeds = torch.cat([vision_embeds, lm_embeds], dim=1)
        if attention_mask is not None:
            vision_attention_mask = torch.ones(*vision_embeds.shape[:2]).to(self.device)
            attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)
        kwargs = dict(
            input_embeds=embeds,
            attention_mask=attention_mask
        )
        return self.decoder(**kwargs).logits

    def compute_loss(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        criterion = nn.CrossEntropyLoss()
        logits = self(img, input_ids[:, :-1], attention_mask[:, :-1])
        vision_pad_size = (logits.shape[0], logits.shape[1]-input_ids.shape[1]+1)
        vision_pad = torch.full(vision_pad_size, self.tokenizer.pad_token_id).to(self.device)
        eos_tensor = torch.full((img.shape[0], 1), self.tokenizer.eos_token_id).to(self.device)
        target = torch.cat([vision_pad, input_ids[:, 1:], eos_tensor], dim=1)
        loss = criterion(logits.permute(0, 2, 1), target)
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
        bleu_meter = self.meters['train/bleu']
        bleu_meter.update(fast_bleu)
        self.log('train/bleu', torch.tensor(bleu_meter.avg), prog_bar=True)
        self.log('train/loss', torch.tensor(loss.item()), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        fast_bleu = self.compute_fast_bleu(batch)
        bleu_meter = self.meters['val/bleu']
        bleu_meter.update(fast_bleu)
        self.log('val/bleu', torch.tensor(bleu_meter.avg), prog_bar=True)
        self.log('val/loss', torch.tensor(loss.item()), prog_bar=True)
        return fast_bleu

    def on_validation_epoch_start(self):
        self.meters['train/bleu'].reset()
        self.meters['val/bleu'].reset()

    @torch.no_grad()
    def infer(self, img, input_ids, max_length, ignore_eos=False):
        assert img.size(0) == 1, 'The inference for batch is not supported yet.'
        output = input_ids
        for i in range(max_length):
            logits = self(img, output.view(1, -1))
            output = logits[0].argmax(dim=0)[-i-1:]
            next_token = output[-1]
            if next_token == self.tokenizer.eos_token_id and not ignore_eos:
                return output
        return output




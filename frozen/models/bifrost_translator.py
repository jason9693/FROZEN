import torch
from einops import repeat
from torch import nn
from transformers import AutoModelForSeq2SeqLM

from frozen.models.base import BiFrostBase
from utils import freeze_module


class BiFrostTranslator(BiFrostBase):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self._set_bleu_metric()
        if self.config.forced_bos_token_id is not None:
            self.bos_token_id = config.forced_bos_token_id
        elif self.config.forced_bos_token is not None:
            self.bos_token_id = self.tokenizer.encode(self.config.forced_bos_token)[0]
        else:
            self.bos_token_id = self.tokenizer.bos_token_id
        with torch.no_grad():
            bos_sep_token_id = torch.tensor([self.bos_token_id, self.tokenizer.sep_token_id]).view(1, -1)
            bos_sep_embed = self.decoder.embed_tokens(bos_sep_token_id)*self.decoder.embed_scale
            bos_embed = bos_sep_embed[:, :1]
            sep_embed = bos_sep_embed[:, 1:]
            bos_embed.requires_grad = False
            sep_embed.requires_grad = False
        self.register_buffer('bos_embed', bos_embed)
        self.register_buffer('sep_embed', sep_embed)

    def _set_decoder(self):
        translator = AutoModelForSeq2SeqLM.from_pretrained(self.config.lm_path)
        self.decoder = translator.get_decoder()
        self.decoder_head = translator.get_output_embeddings()
        freeze_module(self.decoder)
        freeze_module(self.decoder_head)

    @classmethod
    def _get_logits_from_output(cls, output):
        return output

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
    def infer(self, img, max_length, ignore_eos=False):
        decoding_output = torch.tensor([[self.bos_token_id]]).long().cuda()
        for i in range(max_length):
            logits = self(img, decoding_output.view(1, -1))
            decoding_output = logits[0].argmax(dim=0)[-i-1:]
            next_token = decoding_output[-1]
            if next_token == self.tokenizer.eos_token_id and not ignore_eos:
                return decoding_output
        return decoding_output


class BiFrostDistilledTranslator(BiFrostTranslator):
    def _set_decoder(self):
        translator = AutoModelForSeq2SeqLM.from_pretrained(self.config.lm_path)
        self.lm_encoder = translator.get_encoder()
        self.decoder = translator.get_decoder()
        self.decoder_head = translator.get_output_embeddings()
        freeze_module(self.lm_encoder)
        freeze_module(self.decoder)
        freeze_module(self.decoder_head)

    def compute_distill_loss(self, vision_embeds, input_ids):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        criterion = nn.MSELoss()
        logits = self(img, input_ids[:, :-1], attention_mask[:, :-1])
        loss = criterion(logits.view(-1, self.lm_config.vocab_size), input_ids[:, 1:].flatten())
        return loss

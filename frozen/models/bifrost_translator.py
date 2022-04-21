import time

import torch
from einops import repeat
from torch import nn
from transformers import AutoModelForSeq2SeqLM
from transformers.models.fsmt.modeling_fsmt import make_padding_mask, invert_mask, fill_with_neg_inf, triu_onnx

from frozen.models.base import BiFrostBase
from utils import freeze_module


class BiFrostTranslatorBase(BiFrostBase):
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
        self.decoder = translator.model.decoder
        self.lm_config = translator.config
        freeze_module(self.decoder)

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

    def forward_language_decoder(self, vision_embeds, input_ids, attention_mask):
        raise NotImplementedError()

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
        if self.global_step+1 % self.config.logging_interval == 0:
            self.meters['train/bleu'].reset()
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


class BiFrostM2M100(BiFrostTranslatorBase):
    def _set_decoder(self):
        translator = AutoModelForSeq2SeqLM.from_pretrained(self.config.lm_path)
        self.decoder = translator.model.decoder
        self.decoder_head = translator.get_output_embeddings()
        self.lm_config = translator.config
        freeze_module(self.decoder)
        freeze_module(self.decoder_head)

    def forward_language_decoder(self, vision_embeds, input_ids, attention_mask=None):
        kwargs = dict(
            encoder_hidden_states=vision_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.decoder_head(self.decoder(**kwargs).last_hidden_state)


def prepare_fsmt_decoder_inputs(
    config,
    dtype,
    decoder_input_ids=None,
    decoder_padding_mask=None
):
    pad_token_id = config.pad_token_id
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    device = decoder_input_ids.device
    causal_mask = triu_onnx(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1.).to(dtype=dtype, device=device)
    return decoder_padding_mask, causal_mask


class BiFrostFSMT(BiFrostTranslatorBase):
    def forward_language_decoder(self, vision_embeds, input_ids, attention_mask=None):
        decoder_padding_mask, causal_mask = prepare_fsmt_decoder_inputs(
            self.lm_config,
            self.dtype,
            input_ids,
            attention_mask
        )
        kwargs = dict(
            input_ids=input_ids,
            encoder_hidden_states=vision_embeds,
            encoder_padding_mask=None,
            decoder_padding_mask=decoder_padding_mask,
            decoder_causal_mask=causal_mask
        )
        return self.decoder(**kwargs)[0]

    def compute_loss(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        criterion = nn.CrossEntropyLoss()
        logits = self(img, input_ids[:, :-1], attention_mask[:, :-1])
        loss = criterion(logits.view(-1, self.lm_config.tgt_vocab_size), input_ids[:, 1:].flatten())
        return loss


import math

import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import nn
from transformers import AutoModelForSeq2SeqLM

from frozen.models.base import BiFrostBase
from frozen.models.layers import ConvPatchEmbed, PatchEmbedToImage
from utils import freeze_module


class BiBidM2M100(BiFrostBase):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self._set_model()
        self._set_bleu_metric()
        self._register_special_tokens()

    def _register_special_tokens(self):
        if self.config.forced_bos_token_id is not None:
            self.bos_token_id = self.config.forced_bos_token_id
        elif self.config.forced_bos_token is not None:
            self.bos_token_id = self.tokenizer.encode(self.config.forced_bos_token)[0]
        else:
            self.bos_token_id = self.tokenizer.bos_token_id
        with torch.no_grad():
            bos_sep_token_id = torch.tensor([self.bos_token_id, self.tokenizer.sep_token_id]).view(1, -1)
            bos_sep_embed = self.shared(bos_sep_token_id)*self.encoder.embed_scale
            bos_embed = bos_sep_embed[:, :1]
            sep_embed = bos_sep_embed[:, 1:]
            bos_embed.requires_grad = False
            sep_embed.requires_grad = False
        self.register_buffer('bos_embed', bos_embed)
        self.register_buffer('sep_embed', sep_embed)

    @property
    def encoder(self):
        return self.translator.model.encoder

    @property
    def decoder(self):
        return self.translator.model.decoder

    @property
    def shared(self):
        return self.translator.model.shared

    def _set_model(self):
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(self.config.lm_path)
        self.lm_config = self.translator.config
        self.embed_dim = self.lm_config.d_model
        self.to_patch_embed = ConvPatchEmbed(self.embed_dim, self.config.patch_size)
        self.to_reconstructed_img = PatchEmbedToImage(self.embed_dim, self.config.patch_size)
        if self.config.freeze_encoder:
            freeze_module(self.translator.encoder)
        if self.config.freeze_decoder:
            freeze_module(self.translator.decoder)
            freeze_module(self.translator.lm_head)

    def forward(self, input_patches, input_ids, attention_mask=None, task='i2t'):
        if task == 't2t':
            kwargs = dict(
                input_ids=input_ids[:, :-1],
                decoder_input_ids=input_ids[:, :-1],
                decoder_attention_mask=attention_mask[:, :-1]
            )
            return self.translator(**kwargs).logits
        elif task == 'i2t':
            kwargs = dict(
                inputs_embeds=input_patches[:, :-1],
                decoder_input_ids=input_ids[:, :-1],
                decoder_attention_mask=attention_mask[:, :-1]
            )
            return self.translator(**kwargs).logits
        elif task in ('t2i', 't2i+L1'):
            kwargs = dict(
                input_ids=input_ids[:, :-1],
                decoder_inputs_embeds=input_patches[:, :-1]
            )
            return self.translator.model(**kwargs).last_hidden_state
        elif task in ('i2i', 'i2i+L1'):
            kwargs = dict(
                inputs_embeds=input_patches[:, :-1],
                decoder_inputs_embeds=input_patches[:, :-1]
            )
            return self.translator.model(**kwargs).last_hidden_state
        else:
            raise KeyError

    def _generate_input_patches(self, img):
        input_patches = self.to_patch_embed(img)
        bos_embed = repeat(self.bos_embed, '() n d -> b n d', b=img.size(0))
        sep_embed = repeat(self.sep_embed, '() n d -> b n d', b=img.size(0))
        return torch.cat([bos_embed, input_patches, sep_embed], dim=1)

    def compute_loss(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        input_patches = self._generate_input_patches(img)
        h, w = img.shape[2:]
        num_ds = int(math.log2(self.config.patch_size))
        for _ in range(num_ds):
            h = (h+1)//2
            w = (w+1)//2
        losses = dict()
        for task in self.config.translate_tasks:
            output = self(input_patches, input_ids, attention_mask, task)
            if task in ('i2t', 't2t'):
                criterion = nn.CrossEntropyLoss()
                pred = output.view(-1, self.lm_config.vocab_size)
                losses[task] = criterion(pred, input_ids[:, 1:].flatten())
            elif task in ('i2i', 't2i', 'i2i+L1', 't2i+L1'):
                if task in ('i2i+L1', 't2i+L1'):
                    criterion = nn.L1Loss()
                else:
                    criterion = nn.MSELoss()
                # ignore sep token for regression
                pred = rearrange(output[:, :-1], 'b (h w) d -> b d h w', h=h, w=w)
                pred = self.to_reconstructed_img(pred)
                pred = F.interpolate(pred, img.shape[2:], mode='nearest')
                losses[task] = criterion(pred, img)
        return losses

    # compute bleu using teacher-forcing outputs
    @torch.no_grad()
    def compute_fast_bleu(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        input_patches = self._generate_input_patches(img)
        logits = self(input_patches, input_ids, attention_mask, 'i2t')
        output_ids = logits.argmax(dim=-1)
        return self.bleu.score(input_ids[:, 1:], output_ids)

    def training_step(self, batch, batch_idx):
        losses = self.compute_loss(batch)
        fast_bleu = self.compute_fast_bleu(batch)
        bleu_meter = self.meters['train/bleu']
        bleu_meter.update(fast_bleu)
        self.log('train/bleu', torch.tensor(bleu_meter.avg), prog_bar=True)
        total_loss = 0.
        for task, loss in losses.items():
            loss_ratio = self.config.translate_tasks[task]
            self.log(f'train/loss/{task}', torch.tensor((loss_ratio*loss).item()), prog_bar=True)
            total_loss += loss_ratio*loss
        self.log('train/loss/total', torch.tensor(total_loss.item()), prog_bar=True)
        if self.global_step+1 % self.config.logging_interval == 0:
            self.meters['train/bleu'].reset()
        return total_loss

    def validation_step(self, batch, batch_idx):
        losses = self.compute_loss(batch)
        fast_bleu = self.compute_fast_bleu(batch)
        bleu_meter = self.meters['val/bleu']
        bleu_meter.update(fast_bleu)
        self.log('val/bleu', torch.tensor(bleu_meter.avg), prog_bar=True)
        total_loss = 0.
        for task, loss in losses.items():
            loss_ratio = self.config.translate_tasks[task]
            self.log(f'val/loss/{task}', torch.tensor((loss_ratio*loss).item()), prog_bar=True)
            total_loss += loss_ratio*loss
        self.log('val/loss/total', torch.tensor(total_loss.item()), prog_bar=True)
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


class BiBidM2M100WithBEiT(BiFrostBase):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self._set_model()
        self._set_bleu_metric()
        self._register_special_tokens()

    def _register_special_tokens(self):
        if self.config.forced_bos_token_id is not None:
            self.bos_token_id = self.config.forced_bos_token_id
        elif self.config.forced_bos_token is not None:
            self.bos_token_id = self.tokenizer.encode(self.config.forced_bos_token)[0]
        else:
            self.bos_token_id = self.tokenizer.bos_token_id
        with torch.no_grad():
            bos_sep_token_id = torch.tensor([self.bos_token_id, self.tokenizer.sep_token_id]).view(1, -1)
            bos_sep_embed = self.shared(bos_sep_token_id)*self.encoder.embed_scale
            bos_embed = bos_sep_embed[:, :1]
            sep_embed = bos_sep_embed[:, 1:]
            bos_embed.requires_grad = False
            sep_embed.requires_grad = False
        self.register_buffer('bos_embed', bos_embed)
        self.register_buffer('sep_embed', sep_embed)

    @property
    def encoder(self):
        return self.translator.model.encoder

    @property
    def decoder(self):
        return self.translator.model.decoder

    @property
    def shared(self):
        return self.translator.model.shared

    def _set_model(self):
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(self.config.lm_path)
        self.lm_config = self.translator.config
        self.embed_dim = self.lm_config.d_model
        self.to_patch_embed = ConvPatchEmbed(self.embed_dim, self.config.patch_size)
        self.to_reconstructed_img = PatchEmbedToImage(self.embed_dim, self.config.patch_size)
        if self.config.freeze_encoder:
            freeze_module(self.translator.encoder)
        if self.config.freeze_decoder:
            freeze_module(self.translator.decoder)
            freeze_module(self.translator.lm_head)

    def forward(self, input_patches, input_ids, attention_mask=None, task='i2t'):
        if task == 't2t':
            kwargs = dict(
                input_ids=input_ids[:, :-1],
                decoder_input_ids=input_ids[:, :-1],
                decoder_attention_mask=attention_mask[:, :-1]
            )
            return self.translator(**kwargs).logits
        elif task == 'i2t':
            kwargs = dict(
                inputs_embeds=input_patches[:, :-1],
                decoder_input_ids=input_ids[:, :-1],
                decoder_attention_mask=attention_mask[:, :-1]
            )
            return self.translator(**kwargs).logits
        elif task in ('t2i', 't2i+L1'):
            kwargs = dict(
                input_ids=input_ids[:, :-1],
                decoder_inputs_embeds=input_patches[:, :-1]
            )
            return self.translator.model(**kwargs).last_hidden_state
        elif task in ('i2i', 'i2i+L1'):
            kwargs = dict(
                inputs_embeds=input_patches[:, :-1],
                decoder_inputs_embeds=input_patches[:, :-1]
            )
            return self.translator.model(**kwargs).last_hidden_state
        else:
            raise KeyError

    def _generate_input_patches(self, img):
        input_patches = self.to_patch_embed(img)
        bos_embed = repeat(self.bos_embed, '() n d -> b n d', b=img.size(0))
        sep_embed = repeat(self.sep_embed, '() n d -> b n d', b=img.size(0))
        return torch.cat([bos_embed, input_patches, sep_embed], dim=1)

    def compute_loss(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        input_patches = self._generate_input_patches(img)
        h, w = img.shape[2:]
        num_ds = int(math.log2(self.config.patch_size))
        for _ in range(num_ds):
            h = (h+1)//2
            w = (w+1)//2
        losses = dict()
        for task in self.config.translate_tasks:
            output = self(input_patches, input_ids, attention_mask, task)
            if task in ('i2t', 't2t'):
                criterion = nn.CrossEntropyLoss()
                pred = output.view(-1, self.lm_config.vocab_size)
                losses[task] = criterion(pred, input_ids[:, 1:].flatten())
            elif task in ('i2i', 't2i', 'i2i+L1', 't2i+L1'):
                if task in ('i2i+L1', 't2i+L1'):
                    criterion = nn.L1Loss()
                else:
                    criterion = nn.MSELoss()
                # ignore sep token for regression
                pred = rearrange(output[:, :-1], 'b (h w) d -> b d h w', h=h, w=w)
                pred = self.to_reconstructed_img(pred)
                pred = F.interpolate(pred, img.shape[2:], mode='nearest')
                losses[task] = criterion(pred, img)
        return losses

    # compute bleu using teacher-forcing outputs
    @torch.no_grad()
    def compute_fast_bleu(self, batch):
        img = batch['image'][0]
        input_ids = batch['text_ids']
        attention_mask = batch['text_masks']
        input_patches = self._generate_input_patches(img)
        logits = self(input_patches, input_ids, attention_mask, 'i2t')
        output_ids = logits.argmax(dim=-1)
        return self.bleu.score(input_ids[:, 1:], output_ids)

    def training_step(self, batch, batch_idx):
        losses = self.compute_loss(batch)
        fast_bleu = self.compute_fast_bleu(batch)
        bleu_meter = self.meters['train/bleu']
        bleu_meter.update(fast_bleu)
        self.log('train/bleu', torch.tensor(bleu_meter.avg), prog_bar=True)
        total_loss = 0.
        for task, loss in losses.items():
            loss_ratio = self.config.translate_tasks[task]
            self.log(f'train/loss/{task}', torch.tensor((loss_ratio*loss).item()), prog_bar=True)
            total_loss += loss_ratio*loss
        self.log('train/loss/total', torch.tensor(total_loss.item()), prog_bar=True)
        if self.global_step+1 % self.config.logging_interval == 0:
            self.meters['train/bleu'].reset()
        return total_loss

    def validation_step(self, batch, batch_idx):
        losses = self.compute_loss(batch)
        fast_bleu = self.compute_fast_bleu(batch)
        bleu_meter = self.meters['val/bleu']
        bleu_meter.update(fast_bleu)
        self.log('val/bleu', torch.tensor(bleu_meter.avg), prog_bar=True)
        total_loss = 0.
        for task, loss in losses.items():
            loss_ratio = self.config.translate_tasks[task]
            self.log(f'val/loss/{task}', torch.tensor((loss_ratio*loss).item()), prog_bar=True)
            total_loss += loss_ratio*loss
        self.log('val/loss/total', torch.tensor(total_loss.item()), prog_bar=True)
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

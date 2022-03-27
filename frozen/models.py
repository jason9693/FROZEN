import torch
import timm
from transformers import AutoConfig, AutoModelForCausalLM
import pytorch_lightning as pl

from frozen.vision_heads import wrap_vis_encoder, VisionAttentionHead


class GPT2LitFROZEN(pl.LightningModule):
    def __init__(
        self,
        vision_model,
        nlp_model,
        vis_token_proj=None,
        vis_mode='global',
        mlm=False,
        plm=True
    ):
        super().__init__()
        self.lm = nlp_model
        for param in self.lm.parameters():
            param.requires_grad = False
        self.v_encoder = vision_model
        self.mlm = mlm
        self.plm = plm
        self.vis_mode = vis_mode
        self.vis_token_proj = vis_token_proj
        if vis_token_proj is not None:
            self.v_encoder.num_tokens = vis_token_proj.num_output_tokens
    
    def forward(self, img, tokens, **kwargs):
        return self._generate_zero_shot_embeds(self.v_encoder(img), tokens, **kwargs)

    def _generate_zero_shot_embeds(self, vis_embed, tokens, **kwargs):
        input_ids = tokens["input_ids"]
        device = input_ids.device
        if "Model" in type(self.lm).__name__ and "Head" not in type(self.lm).__name__:
            nlp_embed = self.lm.wte(input_ids)
        else:
            nlp_embed = self.lm.transformer.wte(input_ids)
        inputs = {k: v for k, v in tokens.items() if k != "input_ids"}
        if self.vis_token_proj is not None:
            vis_embed = self.vis_token_proj(vis_embed, nlp_embed)
        inputs["inputs_embeds"] = torch.cat([vis_embed, nlp_embed], 1)
        inputs["attention_mask"] = torch.cat(
            [torch.ones(*vis_embed.size()[:2]).to(device), tokens["attention_mask"]], 1)
        lm_output = self.lm(**inputs, **kwargs)
        return lm_output

    # todo
    def _generate_few_shot_embeds(self, vis_embeds, tokens, **kwargs):
        pass

    @classmethod
    def from_pretrained(
        cls,
        hface_path='gpt2',
        vision_path='nf_resnet50',
        pretrained_vision: bool=False,
        emb_key="n_embd",
        vis_mode='global',
        num_global_tokens=2,
        local_output_size=4,
        num_vision_tokens=None,
        **kwargs
    ):
        lm_config = AutoConfig.from_pretrained(hface_path)
        vision = timm.create_model(vision_path, pretrained=pretrained_vision)
        wrap_vis_encoder(
            vision,
            lm_config.to_dict()[emb_key],
            num_global_tokens,
            local_output_size,
            vision_path,
            vis_mode,
            pretrained_vision
        )
        # todo: modify naming to prevent confusing...
        if num_vision_tokens is not None:
            vis_token_proj = VisionAttentionHead(
                dim=lm_config.to_dict()[emb_key],
                num_vision_tokens=vision.num_tokens,
                num_output_tokens=num_vision_tokens
            )
        else:
            vis_token_proj = None
        lm = AutoModelForCausalLM.from_pretrained(hface_path)
        return cls(vision, lm, vis_token_proj, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def set_tokenizer(self, tokenizer, proc_fn=None):
        if proc_fn is not None:
            self.tokenizer = proc_fn(tokenizer)
        else:
            self.tokenizer = tokenizer

    def train_forward(self, img, tokens, loss_fn, target):
        output = self.forward(img, tokens)
        loss = loss_fn(output.logits.transpose(-1, -2), target.to(torch.long))
        return loss

    def encode(self, text):
        return self.tokenizer(text)

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)

    # todo: support directly concatenating attention mask
    def zero_shot_infer(self, img, tokens, max_length):
        assert img.size(0) == 1, 'zero_shot_infer method does not support batch inference.'
        vis_embed = self.v_encoder(img)
        for i in range(max_length):
            output = self._generate_zero_shot_embeds(vis_embed, tokens)
            output = output.logits[0].argmax(dim=-1)
            if output[-1] == self.tokenizer.eos_token_id or i+1 == max_length:
                return output[-i-1:-1]
            input_ids = torch.cat((tokens['input_ids'][0], output[-1:]), dim=0)
            tokens = self.encode(self.decode(input_ids))
            tokens = dict(
                input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0).to(self.device),
                attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0).to(self.device)
            )

    # todo
    def few_shot_infer(self, imgs, tokens, max_length):
        assert imgs.size(0) == 1, 'few_shot_infer method does not support batch inference.'
        pass

    def training_step(self, train_batch, batch_idx):
        img = train_batch["image"][0]
        b_size = img.size()[0]
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        if self.plm:
            tokens = train_batch["text_ids"]
            mask = train_batch["text_masks"]
            tokens = {
                "input_ids": tokens,
                "attention_mask": mask
            }
            eos_tensor = torch.ones(b_size, 1).to(img.device)*self.tokenizer.eos_token_id
            img_pad_tensor = torch.ones(b_size, self.v_encoder.num_tokens).to(img.device)*self.tokenizer.pad_token_id
            target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:, 1:]
            loss = self.train_forward(img, tokens, loss_fn, target)
            self.log('train_plm_loss', loss)
        elif self.mlm:
            tokens = train_batch["text_ids_mlm"]
            mask = train_batch["text_masks"]
            tokens = {
                "input_ids": tokens,
                "attention_mask": mask
            }
            target = train_batch["text_labels_mlm"]
            loss = self.train_forward(img, tokens, loss_fn, target)
            self.log('train_mlm_loss', loss)
        else:
            raise ValueError
        return loss

    def validation_step(self, val_batch, batch_idx):
        img = val_batch["image"][0]
        tokens = val_batch["text_ids"]
        mask = val_batch["text_masks"]
        tokens = {
            "input_ids": tokens,
            "attention_mask": mask
        }
        b_size = img.size()[0]
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        output = self.forward(img, tokens)
        eos_tensor = torch.ones(b_size, 1).to(img.device)*self.tokenizer.eos_token_id
        img_pad_tensor = torch.ones(b_size, self.v_encoder.num_tokens).to(img.device)*self.tokenizer.pad_token_id
        target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:, 1:]
        loss = loss_fn(output.logits.transpose(-1, -2), target.to(torch.long))
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)


class ElectraLitFROZEN(pl.LightningModule):
    def __init__(
        self,
        vision_model,
        nlp_model,
        vis_token_proj=None,
        vis_mode='global',
        mlm=False,
        plm=True
    ):
        super().__init__()
        self.lm = nlp_model
        for param in self.lm.parameters():
            param.requires_grad = False
        for param in self.lm.generator_predictions.parameters():
            param.requires_grad = True
        for param in self.lm.generator_lm_head.parameters():
            param.requires_grad = True
        self.v_encoder = vision_model
        self.vis_mode = vis_mode
        self.mlm = mlm
        self.plm = plm
        self.vis_token_proj = vis_token_proj
        if vis_token_proj is not None:
            self.v_encoder.num_tokens = vis_token_proj.num_output_tokens

    def forward(self, img, tokens, **kwargs):
        return self._generate_zero_shot_embeds(self.v_encoder(img), tokens, **kwargs)

    def _generate_zero_shot_embeds(self, vis_embed, tokens, **kwargs):
        input_ids = tokens["input_ids"]
        device = input_ids.device
        nlp_embed = self.lm.electra.embeddings.word_embeddings(input_ids)
        inputs = {k: v for k, v in tokens.items() if k != "input_ids"}
        if self.vis_token_proj is not None:
            vis_embed = self.vis_token_proj(vis_embed, nlp_embed)
        inputs["inputs_embeds"] = torch.cat([vis_embed, nlp_embed], 1)
        inputs["attention_mask"] = torch.cat(
            [torch.ones(*vis_embed.size()[:2]).to(device), tokens["attention_mask"]], 1)
        lm_output = self.lm(**inputs, **kwargs)
        return lm_output

    # todo
    def _generate_few_shot_embeds(self, vis_embeds, tokens, **kwargs):
        pass

    @classmethod
    def from_pretrained(
        cls,
        hface_path='google/electra-base-discriminator',
        vision_path='nf_resnet50',
        pretrained_vision: bool = False,
        emb_key="embedding_size",
        vis_mode='global',
        num_global_tokens=2,
        local_output_size=4,
        num_vision_tokens=None,
        **kwargs
    ):
        lm_config = AutoConfig.from_pretrained(hface_path)
        vision = timm.create_model(vision_path, pretrained=pretrained_vision)
        wrap_vis_encoder(
            vision,
            lm_config.to_dict()[emb_key],
            num_global_tokens,
            local_output_size,
            vision_path,
            vis_mode,
            pretrained_vision
        )
        # todo: modify naming to prevent confusing...
        if num_vision_tokens is not None:
            vis_token_proj = VisionAttentionHead(
                dim=lm_config.to_dict()[emb_key],
                num_vision_tokens=vision.num_tokens,
                num_output_tokens=num_vision_tokens
            )
        else:
            vis_token_proj = None
        lm = AutoModelForCausalLM.from_pretrained(hface_path)
        return cls(vision, lm, vis_token_proj, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def set_tokenizer(self, tokenizer, proc_fn=None):
        if proc_fn is not None:
            self.tokenizer = proc_fn(tokenizer)
        else:
            self.tokenizer = tokenizer

    def train_forward(self, img, tokens, loss_fn, target):
        output = self.forward(img, tokens)
        loss = loss_fn(output.logits.transpose(-1, -2), target.to(torch.long))
        return loss

    def encode(self, text):
        return self.tokenizer(text)

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)

    # todo: support directly concatenating attention mask
    def zero_shot_infer(self, img, tokens, max_length):
        assert img.size(0) == 1, 'zero_shot_infer method does not support batch inference.'
        vis_embed = self.v_encoder(img)
        for i in range(max_length):
            output = self._generate_zero_shot_embeds(vis_embed, tokens)
            output = output.logits[0].argmax(dim=-1)
            if output[-1] == self.tokenizer.mask_token_id or i+1 == max_length:
                return output[-i-1:-1]
            input_ids = torch.cat((tokens['input_ids'][0], output[-1:]), dim=0)
            tokens = self.encode(self.decode(input_ids))
            tokens = dict(
                input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0).to(self.device),
                attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0).to(self.device)
            )

    # todo
    def few_shot_infer(self, imgs, tokens, max_length):
        assert imgs.size(0) == 1, 'few_shot_infer method does not support batch inference.'
        pass

    def training_step(self, train_batch, batch_idx):
        img = train_batch["image"][0]
        b_size = img.size()[0]
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        if self.plm:
            tokens = train_batch["text_ids"]
            mask = train_batch["text_masks"]
            tokens = {
                "input_ids": tokens,
                "attention_mask": mask
            }
            eos_tensor = torch.ones(b_size, 1).to(img.device)*self.tokenizer.mask_token_id
            img_pad_tensor = torch.ones(b_size, self.v_encoder.num_tokens).to(img.device)*self.tokenizer.pad_token_id
            target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:, 1:]
            loss = self.train_forward(img, tokens, loss_fn, target)
            self.log('train_plm_loss', loss)
        elif self.mlm:
            tokens = train_batch["text_ids_mlm"]
            mask = train_batch["text_masks"]
            tokens = {
                "input_ids": tokens,
                "attention_mask": mask
            }
            target = train_batch["text_labels_mlm"]
            loss = self.train_forward(img, tokens, loss_fn, target)
            self.log('train_mlm_loss', loss)
        else:
            raise ValueError
        return loss

    def validation_step(self, val_batch, batch_idx):
        img = val_batch["image"][0]
        tokens = val_batch["text_ids"]
        mask = val_batch["text_masks"]
        tokens = {
            "input_ids": tokens,
            "attention_mask": mask
        }
        b_size = img.size()[0]
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        output = self.forward(img, tokens)
        eos_tensor = torch.ones(b_size, 1).to(img.device)*self.tokenizer.mask_token_id
        img_pad_tensor = torch.ones(b_size, self.v_encoder.num_tokens).to(img.device)*self.tokenizer.pad_token_id
        target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:, 1:]
        loss = loss_fn(output.logits.transpose(-1, -2), target.to(torch.long))
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)


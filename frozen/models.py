import torch
import timm
from transformers import AutoConfig, AutoModelForCausalLM
import pytorch_lightning as pl

from frozen.vision_heads import wrap_vis_encoder


class LitFROZEN(pl.LightningModule):
    def __init__(
        self,
        vision_model,
        nlp_model,
        vis_mode='global',
        mlm=False,
        plm=True
    ):
        super().__init__()
        self.lm = nlp_model
        for name, param in self.lm.named_parameters():
            param.requires_grad = False
        self.v_encoder = vision_model
        self.mlm = mlm
        self.plm = plm
        self.vis_mode = vis_mode
    
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
        inputs["inputs_embeds"] = torch.cat([vis_embed, nlp_embed], 1)
        inputs["attention_mask"] = torch.cat(
            [torch.ones(*vis_embed.size()[:2]).to(device), tokens["attention_mask"]], 1)
        lm_output = self.lm(**inputs, **kwargs)
        return lm_output

    # todo
    def _generate_few_shot_embeds(self, vis_embeds, tokens, **kwargs):
        input_ids = tokens["input_ids"]  # type: [tokens]
        lengths = [ii.size(1) for ii in input_ids]
        input_ids = torch.cat(input_ids, dim=0)
        device = input_ids.device
        if "Model" in type(self.lm).__name__ and "Head" not in type(self.lm).__name__:
            nlp_embed = self.lm.wte(input_ids)
        else:
            nlp_embed = self.lm.transformer.wte(input_ids)
        inputs = {k: v for k, v in tokens.items() if k != "input_ids"}
        input_embeds = []
        attention_masks = []
        i = 0
        for idx, (length, vis_embed) in enumerate(zip(lengths, vis_embeds)):
            input_embeds.extend([vis_embed, nlp_embed[:, i:length]])
            attention_masks.extend([torch.ones(*vis_embed.size()[:2]).to(device), tokens["attention_mask"][idx]])
            i += length
        inputs["inputs_embeds"] = torch.cat(input_embeds, 1)
        inputs["attention_mask"] = torch.cat(attention_masks, 1)
        lm_output = self.lm(**inputs, **kwargs)
        return lm_output

    @classmethod
    def from_pretrained(
        cls,
        hface_path: str,
        vision_path='nf_resnet50',
        pretrained_vision: bool=False,
        emb_key="n_embd",
        vis_mode='global',
        num_global_tokens=2,
        num_local_tokens=4,
        **kwargs
    ):
        lm_config = AutoConfig.from_pretrained(hface_path)
        vision = timm.create_model(vision_path, pretrained=pretrained_vision)
        wrap_vis_encoder(
            vision,
            lm_config.to_dict()[emb_key],
            num_global_tokens,
            num_local_tokens,
            vision_path,
            vis_mode,
            pretrained_vision
        )
        lm = AutoModelForCausalLM.from_pretrained(hface_path)
        return cls(vision, lm, **kwargs)

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

    def few_shot_infer(self, imgs, tokens, max_length):
        assert imgs.size(0) == 1, 'few_shot_infer method does not support batch inference.'
        vis_embed = self.v_encoder(imgs)
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

    def training_step(self, train_batch, batch_idx):
        img = train_batch["image"][0]
        b_size = img.size()[0]
        loss_fn = torch.nn.CrossEntropyLoss()
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
        loss_fn = torch.nn.CrossEntropyLoss()
        output = self.forward(img, tokens)
        eos_tensor = torch.ones(b_size, 1).to(img.device)*self.tokenizer.eos_token_id
        img_pad_tensor = torch.ones(b_size, self.v_encoder.num_tokens).to(img.device)*self.tokenizer.pad_token_id
        target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:, 1:]
        loss = loss_fn(output.logits.transpose(-1, -2), target.to(torch.long))
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)

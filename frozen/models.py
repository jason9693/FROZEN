import torch
import timm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForPreTraining
import pytorch_lightning as pl

from frozen.vision_heads import wrap_vis_encoder, VisionAttentionHead, LinearPatchEmbed, Stem


def from_pretrained(
    cls,
    hface_path,
    base_lm_class,
    vis_path='nf_resnet50',
    pretrained_vision=False,
    emb_key="n_embd",
    vis_mode='global',
    num_global_tokens=2,
    local_output_size=4,
    num_vis_tokens=None,
    **kwargs
):
    lm_config = AutoConfig.from_pretrained(hface_path)
    embed_size = lm_config.to_dict()[emb_key]
    if vis_path == 'linear':
        vis_model = LinearPatchEmbed()
    elif vis_path == 'conv':
        vis_model = Stem()
    else:
        vis_model = timm.create_model(vis_path, pretrained=pretrained_vision)
    wrap_vis_encoder(
        vis_model,
        embed_size,
        num_global_tokens,
        local_output_size,
        vis_path,
        vis_mode,
        pretrained_vision
    )
    if num_vis_tokens is not None:
        vis_proj_head = VisionAttentionHead(
            dim=embed_size,
            num_input_tokens=vis_model.num_tokens,
            num_output_tokens=num_vis_tokens
        )
    else:
        vis_proj_head = None
    lm = base_lm_class.from_pretrained(hface_path)
    return cls(vis_model, lm, vis_proj_head, vis_mode, **kwargs)


class BiFrostBase(pl.LightningModule):
    def __init__(
        self,
        vis_model,
        nlp_model,
        vis_proj_head=None,
        vis_mode='global'
    ):
        super().__init__()
        self.lm = nlp_model
        for param in self.lm.parameters():
            param.requires_grad = False
        self.v_encoder = vis_model
        self.vis_mode = vis_mode
        self.vis_proj_head = vis_proj_head
        if vis_proj_head is not None:
            self.num_vis_tokens = vis_proj_head.num_output_tokens
        else:
            self.num_vis_tokens = self.v_encoder.num_tokens

    def forward(self, img, tokens, **kwargs):
        return self._compute_output(self.v_encoder(img), tokens, **kwargs)

    def train_forward(self, img, tokens, loss_fn, target):
        output = self.forward(img, tokens)
        loss = loss_fn(output.logits.transpose(-1, -2), target.to(torch.long))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def set_tokenizer(self, tokenizer, proc_fn=None):
        if proc_fn is not None:
            self.tokenizer = proc_fn(tokenizer)
        else:
            self.tokenizer = tokenizer

    def encode(self, text):
        return self.tokenizer(text)

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)

    def _get_text_embeddings(self, input_ids):
        raise NotImplementedError()

    def _compute_output(self, vis_embed, tokens, **kwargs):
        input_ids = tokens["input_ids"]
        device = input_ids.device
        nlp_embed = self._get_text_embeddings(input_ids)
        inputs = {k: v for k, v in tokens.items() if k != "input_ids"}
        if self.vis_proj_head is not None:
            vis_embed = self.vis_proj_head(vis_embed, nlp_embed)
        inputs["inputs_embeds"] = torch.cat([vis_embed, nlp_embed], 1)
        inputs["attention_mask"] = torch.cat(
            [torch.ones(*vis_embed.size()[:2]).to(device), tokens["attention_mask"]], 1)
        lm_output = self.lm(**inputs, **kwargs)
        return lm_output

    @classmethod
    def from_bifrost_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        config = checkpoint['hparams']
        hface_path = config['hface_path']
        vis_path = config['vis_path']
        vis_mode = config['vis_mode']
        emb_key = config['emb_key']
        num_global_tokens = config.get('num_global_tokens', 2)
        local_output_size = config.get('local_output_size', 4)
        num_vis_tokens = config['num_vis_tokens']
        model = cls.from_pretrained(
            cls,
            hface_path,
            vis_path,
            False,
            emb_key,
            vis_mode,
            num_global_tokens,
            local_output_size,
            num_vis_tokens
        )
        model.load_state_dict(state_dict)
        return model, config


class BiFrostCausalLM(BiFrostBase):
    INFERENCE_METHOD = 'plm'
    LANGUAGE_MODEL = 'Causal Language Modeling'

    def training_step(self, train_batch, batch_idx):
        img = train_batch["image"][0]
        b_size = img.size()[0]
        loss_fn = torch.nn.CrossEntropyLoss()
        tokens = train_batch["text_ids"]
        mask = train_batch["text_masks"]
        tokens = {
            "input_ids": tokens,
            "attention_mask": mask
        }
        eos_tensor = torch.ones(b_size, 1).to(img.device)*self.tokenizer.eos_token_id
        img_pad_tensor = -100*torch.ones(b_size, self.num_vis_tokens).to(img.device)
        target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:, 1:]
        loss = self.train_forward(img, tokens, loss_fn, target)
        self.log('train_plm_loss', loss)
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
        eos_tensor = torch.ones(b_size, 1).to(img.device)*self.tokenizer.eos_token_id
        img_pad_tensor = -100*torch.ones(b_size, self.num_vis_tokens).to(img.device)
        target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:, 1:]
        loss = self.train_forward(img, tokens, loss_fn, target)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)


class BiFrostMaskedLM(BiFrostBase):
    INFERENCE_METHOD = 'mlm'
    LANGUAGE_MODEL = 'Masked Language Modeling'

    def training_step(self, train_batch, batch_idx):
        img = train_batch["image"][0]
        loss_fn = torch.nn.CrossEntropyLoss()
        tokens = train_batch["text_ids_mlm"]
        mask = train_batch["text_masks"]
        tokens = {
            "input_ids": tokens,
            "attention_mask": mask
        }
        target = train_batch["text_labels_mlm"]
        img_pad_tensor = -100*torch.ones(img.size(0), self.num_vis_tokens).to(img.device)
        target = torch.cat([img_pad_tensor, target], dim=-1)
        loss = self.train_forward(img, tokens, loss_fn, target)
        self.log('train_mlm_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        img = val_batch["image"][0]
        loss_fn = torch.nn.CrossEntropyLoss()
        tokens = val_batch["text_ids_mlm"]
        mask = val_batch["text_masks"]
        tokens = {
            "input_ids": tokens,
            "attention_mask": mask
        }
        target = val_batch["text_labels_mlm"]
        img_pad_tensor = -100*torch.ones(img.size(0), self.num_vis_tokens).to(img.device)
        target = torch.cat([img_pad_tensor, target], dim=-1)
        loss = self.train_forward(img, tokens, loss_fn, target)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)


class BiFrostGPT2CausalLM(BiFrostCausalLM):
    def _get_text_embeddings(self, input_ids):
        return self.lm.transformer.wte(input_ids)

    @classmethod
    def from_pretrained(
        cls,
        hface_path='gpt2',
        vis_path='nf_resnet50',
        pretrained_vision=False,
        emb_key="n_embd",
        vis_mode='global',
        num_global_tokens=2,
        local_output_size=4,
        num_vis_tokens=None,
        **kwargs
    ):
        return from_pretrained(
            cls,
            hface_path,
            AutoModelForCausalLM,
            vis_path,
            pretrained_vision,
            emb_key,
            vis_mode,
            num_global_tokens,
            local_output_size,
            num_vis_tokens,
            **kwargs
        )


class BiFrostElectraMaskedLM(BiFrostBase):
    def __init__(
        self,
        vis_model,
        nlp_model,
        vis_proj_head=None,
        vis_mode='global'
    ):
        super().__init__(vis_model, nlp_model, vis_proj_head, vis_mode)
        for param in self.lm.generator_predictions.parameters():
            param.requires_grad = True
        for param in self.lm.generator_lm_head.parameters():
            param.requires_grad = True

    def _get_text_embeddings(self, input_ids):
        return self.lm.electra.embeddings.word_embeddings(input_ids)

    @classmethod
    def from_pretrained(
        cls,
        hface_path='google/electra-base-discriminator',
        vis_path='nf_resnet50',
        pretrained_vision=False,
        emb_key="n_embd",
        vis_mode='global',
        num_global_tokens=2,
        local_output_size=4,
        num_vis_tokens=None,
        **kwargs
    ):
        return from_pretrained(
            cls,
            hface_path,
            AutoModelForMaskedLM,
            vis_path,
            pretrained_vision,
            emb_key,
            vis_mode,
            num_global_tokens,
            local_output_size,
            num_vis_tokens,
            **kwargs
        )


class BiFrostBertMaskedLM(BiFrostMaskedLM):
    @classmethod
    def from_pretrained(
        cls,
        hface_path='bert-base-uncased',
        vis_path='nf_resnet50',
        pretrained_vision=False,
        emb_key="n_embd",
        vis_mode='global',
        num_global_tokens=2,
        local_output_size=4,
        num_vis_tokens=None,
        **kwargs
    ):
        return from_pretrained(
            cls,
            hface_path,
            AutoModelForPreTraining,
            vis_path,
            pretrained_vision,
            emb_key,
            vis_mode,
            num_global_tokens,
            local_output_size,
            num_vis_tokens,
            **kwargs
        )


MODEL_FACTORY = {
    'gpt2': BiFrostGPT2CausalLM,
    'electra-base': BiFrostElectraMaskedLM,
    'bert-base': BiFrostBertMaskedLM
}
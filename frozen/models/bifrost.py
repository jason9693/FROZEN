import torch
import timm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForPreTraining
import pytorch_lightning as pl

from frozen.vision_heads import wrap_vis_encoder, LinearPatchEmbed, Stem, VISION_HEAD_FACTORY


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
    interactive_head='concat',
    config_dict=None,
    **kwargs
):
    lm_config = AutoConfig.from_pretrained(hface_path)
    if config_dict is not None:
        lm_config.update(config_dict)
    embed_size = getattr(lm_config, emb_key)
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
    interactive_head_module = VISION_HEAD_FACTORY[interactive_head](
        dim=embed_size,
        num_input_tokens=vis_model.num_tokens,
        num_vis_tokens=num_vis_tokens,
    )
    lm = base_lm_class.from_pretrained(hface_path, config=lm_config)
    return cls(vis_model, lm, interactive_head_module, vis_mode, **kwargs)


class BiFrostBase(pl.LightningModule):
    def __init__(
        self,
        vis_model,
        nlp_model,
        interactive_head,
        vis_mode='global'
    ):
        super().__init__()
        self.lm = nlp_model
        for param in self.lm.parameters():
            param.requires_grad = False
        self.vis_model = vis_model
        self.vis_mode = vis_mode
        self.interactive_head = interactive_head
        self.num_vis_tokens = interactive_head.num_vis_tokens

    def forward(self, img, tokens, **kwargs):
        return self.compute_output(self.vis_model(img), tokens, **kwargs)

    def train_forward(self, img, tokens, loss_fn, target):
        output = self.forward(img, tokens)
        loss = loss_fn(self._get_logits(output).transpose(-1, -2), target.to(torch.long))
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

    def _get_logits(self, output):
        return output.logits

    def compute_output(self, vis_embed, tokens, **kwargs):
        input_ids = tokens["input_ids"]
        nlp_embed = self._get_text_embeddings(input_ids)
        inputs = {k: v for k, v in tokens.items() if k != "input_ids"}
        embed, attn_mask = self.interactive_head(vis_embed, nlp_embed, tokens['attention_mask'])
        inputs["inputs_embeds"] = embed
        inputs["attention_mask"] = attn_mask
        lm_output = self.lm(**inputs, **kwargs)
        return lm_output

    @classmethod
    def from_bifrost_checkpoint(cls, checkpoint_path, config_dict=None):
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        config = checkpoint['hyper_parameters']
        hface_path = config['hface_path']
        vis_path = config['vis_path']
        vis_mode = config['vis_mode']
        interactive_head = config['interactive_head']
        emb_key = config['emb_key']
        num_global_tokens = config.get('num_global_tokens', 2)
        local_output_size = config.get('local_output_size', 4)
        num_vis_tokens = config['num_vis_tokens']
        model = MODEL_FACTORY[config['lm_mode']].from_pretrained(
            hface_path,
            vis_path,
            False,
            emb_key,
            vis_mode,
            num_global_tokens,
            local_output_size,
            num_vis_tokens,
            interactive_head,
            config_dict
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
        target = tokens["input_ids"]
        img_pad_tensor = self.interactive_head.get_vis_label(img)
        if img_pad_tensor is not None:
            target = torch.cat([img_pad_tensor, target], dim=-1)
        target = torch.cat([target, eos_tensor], dim=-1)[:, 1:]
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
        target = tokens["input_ids"]
        img_pad_tensor = self.interactive_head.get_vis_label(img)
        if img_pad_tensor is not None:
            target = torch.cat([img_pad_tensor, target], dim=-1)
        target = torch.cat([target, eos_tensor], dim=-1)[:, 1:]
        loss = self.train_forward(img, tokens, loss_fn, target)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)

    def infer(self, img, tokens, max_length, ignore_eos_token_id):
        assert img.size(0) == 1, 'infer method does not support batch inference.'
        vis_embed = self.vis_model(img)
        result = None
        for i in range(max_length):
            output = self.compute_output(vis_embed, tokens)
            attentions = output.attentions
            output = self._get_logits(output)[0].argmax(dim=-1)
            result = (output[-i-1:-1], attentions)
            if output[-1] == self.tokenizer.eos_token_id and not ignore_eos_token_id:
                return result
            input_ids = torch.cat((tokens['input_ids'][0], output[-1:]), dim=0)
            tokens = dict(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                attention_mask=torch.ones_like(input_ids).unsqueeze(0).to(self.device)
            )
        return result


class BiFrostMaskedLM(BiFrostBase):
    INFERENCE_METHOD = 'mlm'
    LANGUAGE_MODEL = 'Masked Language Modeling'

    def compute_output(self, vis_embed, tokens, **kwargs):
        input_ids = tokens["input_ids"]
        nlp_embed = self._get_text_embeddings(input_ids)
        inputs = {k: v for k, v in tokens.items() if k != "input_ids"}
        embed, _ = self.interactive_head(vis_embed, nlp_embed, tokens['attention_mask'])
        inputs["inputs_embeds"] = embed
        inputs["attention_mask"] = None
        lm_output = self.lm(**inputs, **kwargs)
        return lm_output

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
        img_pad_tensor = self.interactive_head.get_vis_label(img)
        if img_pad_tensor is not None:
            target = torch.cat([target[:, :1], img_pad_tensor, target[:, 1:]], dim=-1)
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
        img_pad_tensor = self.interactive_head.get_vis_label(img)
        if img_pad_tensor is not None:
            target = torch.cat([target[:, :1], img_pad_tensor, target[:, 1:]], dim=-1)
        loss = self.train_forward(img, tokens, loss_fn, target)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        return self.validation_step(test_batch, batch_idx)

    def infer(self, img, tokens):
        assert img.size(0) == 1, 'infer method does not support batch inference.'
        vis_embed = self.vis_model(img)
        output = self.compute_output(vis_embed, tokens)
        attentions = output.attentions
        output = self._get_logits(output)[0].argmax(dim=-1)
        return output[self.num_vis_tokens:], attentions


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
        interactive_head='concat',
        config_dict=None,
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
            interactive_head,
            config_dict,
            **kwargs
        )


class BiFrostElectraMaskedLM(BiFrostMaskedLM):
    def __init__(
        self,
        vis_model,
        nlp_model,
        interactive_head=None,
        vis_mode='global'
    ):
        super().__init__(vis_model, nlp_model, interactive_head, vis_mode)
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
        interactive_head='concat',
        config_dict=None,
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
            interactive_head,
            config_dict,
            **kwargs
        )


class BiFrostBertMaskedLM(BiFrostMaskedLM):
    def _get_text_embeddings(self, input_ids):
        return self.lm.bert.embeddings.word_embeddings(input_ids)

    def _get_logits(self, output):
        return output.prediction_logits

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
        interactive_head='concat',
        config_dict=None,
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
            interactive_head,
            config_dict,
            **kwargs
        )


MODEL_FACTORY = {
    'gpt2': BiFrostGPT2CausalLM,
    'electra-base': BiFrostElectraMaskedLM,
    'bert-base': BiFrostBertMaskedLM
}
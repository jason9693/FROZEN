import torch
import timm
from transformers import AutoConfig, AutoModelForCausalLM
import pytorch_lightning as pl

class FrozenModel(torch.nn.Module):
    def __init__(self, vision_model, nlp_model):
        super().__init__()
        self.lm = nlp_model
        for name, param in self.lm.named_parameters():
            param.requires_grad = False
        self.v_encoder = vision_model
    
    def forward(self, img, tokens, **kwargs):
        vis_embed = self.v_encoder(img)
        vis_embed_shape = vis_embed.size()
        vis_embed = vis_embed.reshape([vis_embed_shape[0], 2, int(vis_embed_shape[-1]/2)])
        
        input_ids = tokens["input_ids"]
        
        if "Model" in type(self.lm).__name__ and "Head" not in type(self.lm).__name__:
            nlp_embed = self.lm.wte(input_ids)
        else:
            nlp_embed = self.lm.transformer.wte(input_ids)
        
        inputs = {k: v for k,v in tokens.items() if k != "input_ids"}
        inputs["inputs_embeds"] = torch.cat([vis_embed, nlp_embed], 1)
        inputs["attention_mask"] = torch.cat([torch.ones(vis_embed_shape[0], 2), tokens["attention_mask"]], 1)
        
        lm_output = self.lm(**inputs, **kwargs)
        
        return lm_output
    
    @classmethod
    def from_pretrained(cls, hface_path: str, pretrained_vision: bool=False):
        lm_config = AutoConfig.from_pretrained(hface_path)
        
        vision = timm.create_model('nf_resnet50', pretrained=pretrained_vision)
        vision.head.fc = torch.nn.Linear(2048, lm_config.n_embd*2) # for prefix embedding
        
        lm = AutoModelForCausalLM.from_pretrained(hface_path)
        return cls(vision, lm)
    
    @classmethod
    def from_trained(cls, path: str):
        pass


class LitFROZEN(pl.LightningModule):
    def __init__(self, vision_model, nlp_model, mlm=False, plm=True):
        super().__init__()
        self.lm = nlp_model
        for name, param in self.lm.named_parameters():
            param.requires_grad = False
        self.v_encoder = vision_model
        self.mlm = mlm
        self.plm = plm
    
    def forward(self, img, tokens, **kwargs):
        vis_embed = self.v_encoder(img)
        vis_embed_shape = vis_embed.size()
        vis_embed = vis_embed.reshape([vis_embed_shape[0], 2, int(vis_embed_shape[-1]/2)])
        
        input_ids = tokens["input_ids"]
        device = input_ids.device
        
        if "Model" in type(self.lm).__name__ and "Head" not in type(self.lm).__name__:
            nlp_embed = self.lm.wte(input_ids)
        else:
            nlp_embed = self.lm.transformer.wte(input_ids)
        
        inputs = {k: v for k,v in tokens.items() if k != "input_ids"}
        inputs["inputs_embeds"] = torch.cat([vis_embed, nlp_embed], 1)
        inputs["attention_mask"] = torch.cat([torch.ones(vis_embed_shape[0], 2).to(device), tokens["attention_mask"]], 1)
        
        lm_output = self.lm(**inputs, **kwargs)
        
        return lm_output
    
    @classmethod
    def from_pretrained(cls, hface_path: str, vision_path='nf_resnet50', pretrained_vision: bool=False):
        lm_config = AutoConfig.from_pretrained(hface_path)
        
        vision = timm.create_model(vision_path, pretrained=pretrained_vision)
        vision.head.fc = torch.nn.Linear(2048, lm_config.n_embd*2) # for prefix embedding
        
        lm = AutoModelForCausalLM.from_pretrained(hface_path)

        return cls(vision, lm)

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
        loss = loss_fn(output.logits.transpose(-1,-2), target.to(torch.long))
        return loss

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
            eos_tensor = torch.ones(b_size,1).to(img.device) * self.tokenizer.eos_token_id
            img_pad_tensor = torch.ones(b_size,2).to(img.device) * self.tokenizer.pad_token_id

            target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:,1:]
            loss = self.train_forward(img, tokens, loss_fn, target)
            self.log('train_plm_loss', loss)

        if self.mlm:
            tokens = train_batch["text_ids_mlm"]
            mask = train_batch["text_masks"]
            tokens = {
                "input_ids": tokens,
                "attention_mask": mask
            }
            target = train_batch["text_labels_mlm"]
            loss = self.train_forward(img, tokens, loss_fn, target)
            self.log('train_mlm_loss', loss)
            
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

        eos_tensor = torch.ones(b_size,1).to(img.device) * self.tokenizer.eos_token_id
        img_pad_tensor = torch.ones(b_size,2).to(img.device) * self.tokenizer.pad_token_id

        target = torch.cat([img_pad_tensor, tokens["input_ids"], eos_tensor], -1)[:,1:]
        output = self.forward(img, tokens)
        loss = loss_fn(output.logits.transpose(-1,-2), target.to(torch.long))

        self.log('val_loss', loss)
        return loss

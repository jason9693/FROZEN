import torch
import timm
from transformers import AutoConfig, AutoModelForCausalLM


class FrozenModel(torch.nn.Module):
    def __init__(self, vision_model, nlp_model):
        super().__init__()
        self.lm = nlp_model
        self.lm.eval()
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
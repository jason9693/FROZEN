import torch
import unittest
from frozen.models import FrozenModel, LitFROZEN
from transformers import AutoTokenizer

model_names = ["gpt2", "bert-base-uncased"]
b=3
l=12
c=3
d=256

def setup(model_name, **kwargs):
    model = LitFROZEN.from_pretrained(model_name, pretrained_vision=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model.set_tokenizer(tokenizer)
    return model, tokenizer
mok_img = torch.rand(b,c,d,d)

class TestLitFROZEN(unittest.TestCase):
    def setup(model_name, **kwargs):
        model = LitFROZEN.from_pretrained(model_name, pretrained_vision=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        model.set_tokenizer(tokenizer)
        return model, tokenizer
    def test_load(self):
        model_names = ["gpt2", "bert-base-uncased"]
        for model_name in model_names:
            model, tokenizer = setup(model_name)
        return
    
    def test_forward(self):
        model_names = ["gpt2", "bert-base-uncased"]
        for model_name in model_names:
            model, tokenizer = setup(model_name)
            model.forward()
        return

if __name__ == "__main__":
    unittest.main()
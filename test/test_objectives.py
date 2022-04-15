import unittest
import numpy as np
from frozen.objectives import *

class TestBLEU(unittest.TestCase):
    def test_score(self):
        refs = ["hello everyone", "my name is kevin."]
        preds = ["hello mike", "my name is kelvin."]

        model_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        metric_tok = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

        refs_tokens = model_tok(refs, return_tensors='pt', padding=True)["input_ids"]
        preds_tokens = model_tok(preds, return_tensors='pt', padding=True)["input_ids"]

        scorer = SacreBLEU(
            2, 
            model_tok,
            metric_tok
        )
        score = scorer.score(preds_tokens, refs_tokens)
        assert 0.534 < score < 0.535
    
    def test_initialize(self):
        scorer = SacreBLEU(
            2, 
            model_tok,
            metric_tok
        )

if __name__ == "__main__":
    unittest.main()
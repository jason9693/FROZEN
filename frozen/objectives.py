import torch
from torchtext.data.metrics import bleu_score

from transformers import AutoTokenizer

import ctypes
from abc import ABC, abstractmethod


class BleuStat(ctypes.Structure):
    _fields_ = [
        ("reflen", ctypes.c_size_t),
        ("predlen", ctypes.c_size_t),
        ("match1", ctypes.c_size_t),
        ("count1", ctypes.c_size_t),
        ("match2", ctypes.c_size_t),
        ("count2", ctypes.c_size_t),
        ("match3", ctypes.c_size_t),
        ("count3", ctypes.c_size_t),
        ("match4", ctypes.c_size_t),
        ("count4", ctypes.c_size_t),
    ]


class BaseScorer(ABC):
    # def add_string(self, ref, pred):
    #     self.ref.append(ref)
    #     self.pred.append(pred)

    @abstractmethod
    def score(self, refs, preds) -> float:
        pass
        

class SacreBLEU(BaseScorer):
    def __init__(self, n_gram, model_tokenizer, metric_tokenizer):
        self.n_gram = n_gram
        self.model_tok = model_tokenizer
        self.metric_tok = metric_tokenizer

    def score(self, refs: torch.Tensor, preds: torch.Tensor):
        decoded_refs = self.model_tok.batch_decode(refs, skip_special_tokens=True)
        decoded_preds = self.model_tok.batch_decode(preds, skip_special_tokens=True)
        
        refs_tokens = [
            [self.metric_tok.tokenize(rs)] for rs in decoded_refs
        ]
        preds_tokens = [
            self.metric_tok.tokenize(rs) for rs in decoded_preds
        ]
        
        # print(refs_tokens)
        # print(preds_tokens)

        return bleu_score(preds_tokens, refs_tokens, max_n=self.n_gram, weights=[1./self.n_gram]*self.n_gram)


if __name__ == "__main__":
    a = ["hello everyone", "my name is kevin."]
    b = ["hello mike", "my name is kelvin."]
    model_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    metric_tok = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

    a = model_tok(a, return_tensors='pt', padding=True)["input_ids"]
    b = model_tok(b, return_tensors='pt', padding=True)["input_ids"]

    scorer = SacreBLEU(
        2, 
        model_tok,
        metric_tok
    )

    print(scorer.score(a, b))
        
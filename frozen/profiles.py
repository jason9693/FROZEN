from frozen.models import GPT2LitFROZEN, ElectraMaskedLitFROZEN, BertMaskedLitFROZEN

VIS_MODE_DICT = {
    'duel': {'mode': 'duel', 'demo_path': 'duel'},
    'global': {'mode': 'global', 'demo_path': 'global'},
    'local': {'mode': 'local', 'demo_path': 'local'}
}
LM_MODE_DICT = {
    'gpt2': {
        'lm': 'gpt2',
        'emb_key': 'n_embd',
        'pad_token': '<|endoftext|>',
        'demo_path': 'gpt2',
        'inference_method': 'plm',
        'cls': GPT2LitFROZEN
    },
    'electra-base': {
        'lm': 'google/electra-base-discriminator',
        'emb_key': 'embedding_size',
        'demo_path': 'electra_base',
        'inference_method': 'mlm',
        'cls': ElectraMaskedLitFROZEN
    },
    'bert-base': {
        'lm': 'bert-base-uncased',
        'emb_key': 'hidden_size',
        'demo_path': 'bert_base',
        'inference_method': 'mlm',
        'cls': BertMaskedLitFROZEN
    }
}
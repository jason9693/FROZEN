from frozen.models.bibid import BiBidM2M100
from frozen.models.bifrost_causal_lm import BiFrostCausalLMBase
from frozen.models.bifrost_masked_lm import BiFrostMaskedLMBase
from frozen.models.bifrost_pretrain import BiFrostForPreTrainingBase


# TODO
def set_model_cls_from_config(config):
    if config.lm_task == 'translate':
        if config.lm_path == 'facebook/m2m100_418M':
            return BiBidM2M100
    elif config.lm_task == 'mlm':
        return BiFrostMaskedLMBase
    elif config.lm_task == 'causal':
        return BiFrostCausalLMBase
    elif config.lm_task == 'pretrain':
        return BiFrostForPreTrainingBase
    else:
        raise KeyError


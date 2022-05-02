import warnings

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from timm.utils import AverageMeter
from transformers import AutoTokenizer

from frozen.models.vision_models import PretrainedVisionEncoder
from frozen.objectives import SacreBLEU


class BiFrostBase(pl.LightningModule):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    # for convenience
    def _set_bleu_metric(self):
        default_bleu_kwargs = dict(n_gram=2)
        if self.config.bleu_conf is not None:
            default_bleu_kwargs.update(self.config.bleu_conf)
        bleu_tokenizer = AutoTokenizer.from_pretrained(self.config.bleu_tokenizer_path)
        self.bleu = SacreBLEU(
            model_tokenizer=self.tokenizer,
            metric_tokenizer=bleu_tokenizer,
            **default_bleu_kwargs
        )
        self.meters = {'train/bleu': AverageMeter(), 'val/bleu': AverageMeter()}

    def configure_optimizers(self):
        if self.config.opt_type == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.betas[0], self.config.betas[1])
            )
        elif self.config.opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                nesterov=self.config.nesterov
            )
        else:
            raise KeyError
        if self.config.sched_type is None:
            return optimizer
        else:
            if self.config.sched_type == 'multistep':
                if self.config.sched_milestones is None:
                    raise AttributeError
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=self.config.multistep_milestones,
                    gamma=self.config.multistep_decay_rate
                )
            elif self.config.sched_type == 'cos':
                num_steps = self.config.max_epochs/self.config.sched_freq
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)
            else:
                raise KeyError
            lr_scheduler = dict(
                scheduler=scheduler,
                interval=self.config.sched_interval,
                frequency=self.config.sched_freq
            )
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def on_save_checkpoint(self, checkpoint):
        config = OmegaConf.to_object(self.config)
        checkpoint['config'] = config


class BiFrostVisionEncBase(BiFrostBase):
    def __init__(self, config, tokenizer=None):
        super().__init__(config, tokenizer)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.lm_path)
        self._set_decoder()
        self.embed_dim = self.lm_config.d_model
        self._set_encoder()

    def _set_encoder(self):
        self.encoder = PretrainedVisionEncoder.from_pretrained(
            self.config.encoder_path,
            pretrained=self.config.use_pretrained_encoder
        )
        if self.config.freeze_encoder:
            if not self.config.use_pretrained_encoder:
                warnings.warn('use_pretrained_encoder must be true for fine-tuning; It will be ignored.')
            else:
                self.encoder.freeze(self.config.num_frozen_stages)

    def _set_decoder(self):
        raise NotImplementedError()


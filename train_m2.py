from copy import deepcopy
import os

import pytorch_lightning as pl

from frozen.config import ex_nmt
from frozen.datamodules.multitask_datamodule import MTDataModule
from frozen.models import set_model_cls_from_config

from omegaconf import OmegaConf

PYTHON_PATH = os.path.abspath('/')


@ex_nmt.automain
def main(_config):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    print(_config)
    config_clone = deepcopy(_config)
    config = OmegaConf.create(config_clone)
    if config.num_nodes > 1:
        dist_backend = 'horovod'
        pl_num_nodes = 1
        pl_num_gpus = 1
    else:
        dist_backend = 'ddp'
        pl_num_nodes = config.num_nodes
        pl_num_gpus = config.num_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{','.join([str(g) for g in range(config.num_gpus)])}"
    pl.seed_everything(config.seed)
    dm = MTDataModule(config_clone, dist=True)
    dm.prepare_data_per_node = False
    model_cls = set_model_cls_from_config(config)
    model = model_cls(config, dm.tokenizer)
    for k, v in config_clone.items():
        model.hparams[k] = v
    exp_name = f'M2'
    if config.ex_tag:
        exp_name = f'{exp_name}_{config.ex_tag}'

    os.makedirs(config.log_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/bleu",
        mode="max",
        save_last=True,
        filename=exp_name,
        dirpath=config.checkpoint_dirpath
    )
    logger_name = f'{exp_name}_seed{config.seed}'
    if config.load_path:
        logger_name += f'_from_{config.load_path.split("/")[-1][:-5]}'
    tb_logger = pl.loggers.TensorBoardLogger(
        config.log_dir,
        name=logger_name
    )
    callbacks = [checkpoint_callback]

    num_gpus = config.num_gpus if isinstance(config.num_gpus, int) else len(config.num_gpus)
    grad_steps = config.batch_size//(config.per_gpu_batchsize*num_gpus*config.num_nodes)
    grad_steps = max(grad_steps, 1)  # handle when grad_steps=0
    max_steps = config.max_steps or -1
    max_epochs = config.max_epochs if max_steps == -1 else 1000

    trainer = pl.Trainer(
        gpus=pl_num_gpus,
        num_nodes=pl_num_nodes,
        precision=config.precision,
        strategy=dist_backend,
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=tb_logger,
        replace_sampler_ddp=True,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=config.logging_interval,
        flush_logs_every_n_steps=config.logging_interval,
        resume_from_checkpoint=config.resume_from,
        weights_summary="top",
        fast_dev_run=config.fast_dev_run,
        val_check_interval=config.val_check_interval,
        amp_level=config.amp_level,
        amp_backend='apex',
        sync_batchnorm=True
    )
    if not config.test_only:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
        print("test finished")
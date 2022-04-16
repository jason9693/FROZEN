import copy
import os

import pytorch_lightning as pl

from frozen.config import ex_m2
from frozen.datamodules.multitask_datamodule import MTDataModule
from frozen.models.m2 import ModalityTranslator

PYTHON_PATH = os.path.abspath('/')


@ex_m2.automain
def main(
    seed,
    ex_tag,
    log_dir,
    load_path,
    num_gpus,
    batch_size,
    per_gpu_batchsize,
    num_nodes,
    max_steps,
    precision,
    max_epoch,
    resume_from,
    fast_dev_run,
    val_check_interval,
    amp_level,
    test_only,
    checkpoint_dirpath,
    use_pretrained_vision_encoder,
    freeze_vision_encoder,
    num_frozen_stages,
    opt_type,
    sched_type,
    sched_milestones,
    _config
):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if num_nodes > 1:
        dist_backend = 'horovod'
        nodes = 1
        gpus = 1
    else:
        dist_backend = 'ddp'
        nodes = num_nodes
        gpus = num_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{','.join([str(g) for g in range(num_gpus)])}"
    print(_config)
    config_clone = copy.deepcopy(_config)
    config_clone['tokenizer'] = "facebook/m2m100_418M"
    pl.seed_everything(seed)
    dm = MTDataModule(config_clone, dist=True)
    dm.prepare_data_per_node = False
    model = ModalityTranslator(
        tokenizer=dm.tokenizer,
        opt_type=opt_type,
        sched_type=sched_type,
        sched_milestones=sched_milestones,
        use_pretrained_vision_encoder=use_pretrained_vision_encoder,
        freeze_vision_encoder=freeze_vision_encoder,
        num_frozen_stages=num_frozen_stages
    )
    for k, v in config_clone.items():
        model.hparams[k] = v
    file_name = exp_name = f'M2'
    if ex_tag:
        exp_name = f'{exp_name}_{ex_tag}'

    os.makedirs(log_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="m2/val_bleu",
        mode="max",
        save_last=True,
        filename=file_name,
        dirpath=checkpoint_dirpath
    )
    logger_name = f'{exp_name}_seed{seed}'
    if load_path:
        logger_name += f'_from_{load_path.split("/")[-1][:-5]}'
    tb_logger = pl.loggers.TensorBoardLogger(
        log_dir,
        name=logger_name
    )
    callbacks = [checkpoint_callback]

    num_gpus = num_gpus if isinstance(num_gpus, int) else len(num_gpus)
    grad_steps = batch_size//(per_gpu_batchsize*num_gpus*num_nodes)
    grad_steps = max(grad_steps, 1)  # handle when grad_steps=0
    max_steps = max_steps if max_steps is not None else -1

    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=nodes,
        precision=precision,
        strategy=dist_backend,
        benchmark=True,
        deterministic=True,
        max_epochs=max_epoch if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=tb_logger,
        replace_sampler_ddp=True,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=resume_from,
        weights_summary="top",
        fast_dev_run=fast_dev_run,
        val_check_interval=val_check_interval,
        amp_level=amp_level,
        amp_backend='apex',
    )
    if not test_only:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
        print("test finished")
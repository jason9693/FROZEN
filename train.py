import copy
import os

import pytorch_lightning as pl

from frozen.config import ex
from frozen.datamodules.multitask_datamodule import MTDataModule
from frozen.models import MODEL_FACTORY

PYTHON_PATH = os.path.abspath('./')

def _get_model(lm_mode, hface_path, emb_key, vis_path, vis_mode, num_vis_tokens):
    model = MODEL_FACTORY[lm_mode].from_pretrained(
        hface_path, vis_path=vis_path, emb_key=emb_key, vis_mode=vis_mode, num_vis_tokens=num_vis_tokens)
    return model


@ex.automain
def main(
    seed,
    lm_mode,
    hface_path,
    emb_key,
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
    vis_path,
    vis_mode,
    pretrained_vision,
    num_vis_tokens,
    checkpoint_dirpath,
    _config
):
    if num_nodes > 1:
        dist_backend = 'horovod'
        gpus = 1
    else:
        dist_backend = 'ddp'
        gpus = num_gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = f"{','.join([str(g) for g in range(num_gpus)])}"
    print(_config)
    config_clone = copy.deepcopy(_config)
    pl.seed_everything(seed)
    dm = MTDataModule(config_clone, dist=True)
    path_key = f'{lm_mode}_{vis_path}_{vis_mode}'
    if num_vis_tokens is not None:
        path_key = path_key+'_vh'
    if pretrained_vision:
        path_key = path_key+'_ft'
    model = _get_model(lm_mode, hface_path, emb_key, vis_path, vis_mode, num_vis_tokens)
    for k, v in config_clone.items():
        model.hparams[k] = v
    file_name = exp_name = f'BiFrost_{path_key}'
    if ex_tag:
        exp_name = f'{exp_name}_{ex_tag}'
    model.set_tokenizer(dm.tokenizer)

    os.makedirs(log_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
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
    max_steps = max_steps if max_steps is not None else None

    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=num_nodes,
        precision=precision,
        accelerator=dist_backend,
        benchmark=True,
        deterministic=True,
        max_epochs=max_epoch if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=tb_logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=resume_from,
        weights_summary="top",
        fast_dev_run=fast_dev_run,
        val_check_interval=val_check_interval,
        amp_level=amp_level
    )
    if not test_only:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
        print("test finished")
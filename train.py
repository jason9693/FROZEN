import os
import copy
import pytorch_lightning as pl

from frozen.config import ex
from frozen.models import LitFROZEN
from frozen.datamodules.multitask_datamodule import MTDataModule

import pdb


@ex.automain
def main(
    seed,
    lm,
    emb_key,
    exp_name,
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
    vis_mode,
    _config
):
    # _config = copy.deepcopy(_config)
    pl.seed_everything(seed)
    dm = MTDataModule(_config, dist=True)

    model = LitFROZEN.from_pretrained(lm, emb_key=emb_key, vis_mode=vis_mode)
    model.set_tokenizer(dm.tokenizer)
    exp_name = f'{exp_name}'

    os.makedirs(log_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        log_dir,
        name=f'{exp_name}_seed{seed}_from_{load_path.split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        num_gpus
        if isinstance(num_gpus, int)
        else len(num_gpus)
    )

    grad_steps = batch_size // (
        per_gpu_batchsize * num_gpus * num_nodes
    )
    grad_steps = max(grad_steps, 1)

    max_steps = max_steps if max_steps is not None else None

    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=num_nodes,
        precision=precision,
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=max_epoch if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
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
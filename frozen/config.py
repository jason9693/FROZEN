import os

from sacred import Experiment

ex = Experiment("FROZEN")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = ""
    seed = 0
    datasets = ["vqa"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    lm = "gpt2"
    # lm = "gpt-neo-1.3B"
    vqav2_label_size = 3129
    max_text_len = 40
    # vocab_size = 30522
    whole_word_masking = False
    # mlm_prob = 0.15
    draw_false_text = 0

    # Tokenizer setting
    if 'gpt' in lm:
        pad_token = '<|endoftext|>'
    tokenizer = lm
    emb_key = "n_embd"

    # Optimizer Setting
    optim_type = "adam"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = f"{os.getcwd()}/dataset/coco/"
    log_dir = "result"
    per_gpu_batchsize = 16  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32
    amp_level = "O1"

    vis_mode = "global"
    v_encoder = "nf_resnet50"
    pretrained_vision = False
    num_vision_tokens = None

@ex.named_config
def task_finetune_gpt2():
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

@ex.named_config
def task_finetune_electra():
    lm = "google/electra-small-discriminator"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

    ## huggingface lm config
    emb_key = "embedding_size"
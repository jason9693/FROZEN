import os

from sacred import Experiment

ex = Experiment("FROZEN")
ex_m2 = Experiment("M2")


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
    python_path = os.path.abspath('./')
    ex_tag = ""
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
    # lm = "gpt-neo-1.3B"
    vqav2_label_size = 3129
    max_text_len = 40
    # vocab_size = 30522
    whole_word_masking = False
    # mlm_prob = 0.15
    draw_false_text = 0

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
    data_root = f"/project/arrows"  # path of arrow files
    log_dir = "result"
    per_gpu_batchsize = 16  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32
    amp_level = "O1"

    vis_mode = "global"
    vis_path = "nf_resnet50"
    interactive_head = "concat"
    pretrained_vision = False
    num_vis_tokens = None
    checkpoint_dirpath = '/nas/po.ai'

@ex.named_config
def task_finetune_gpt2():
    lm_mode = 'gpt2'
    hface_path = 'gpt2'
    tokenizer = hface_path
    datasets = ["coco", "gcc"]
    pad_token = "<|endoftext|>"
    loss_names = _loss_names({"itm": 1})
    batch_size = 512
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

@ex.named_config
def task_finetune_electra():
    lm_mode = 'electra-base'
    hface_path = 'google/electra-base-discriminator'
    tokenizer = hface_path
    datasets = ["coco", "gcc"]
    loss_names = _loss_names({"mlm": 1})
    batch_size = 512
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10
    mlm_prob = 0.15

    ## huggingface lm config
    emb_key = "embedding_size"

@ex.named_config
def task_finetune_bert_base():
    lm_mode = "bert-base"
    hface_path = 'bert-base-uncased'
    tokenizer = hface_path
    datasets = ["coco", "gcc"]
    loss_names = _loss_names({"mlm": 1})
    batch_size = 512
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10
    mlm_prob = 0.15

    ## huggingface lm config
    emb_key = "hidden_size"


@ex_m2.config
def config():
    ex_tag = ""
    seed = 0
    datasets = ["coco", "gcc"]
    loss_names = _loss_names({"itm": 1})

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    image_only = False

    # Text Setting
    # lm = "gpt-neo-1.3B"
    vqav2_label_size = 3129
    max_text_len = 40
    # vocab_size = 30522
    whole_word_masking = False
    # mlm_prob = 0.15
    draw_false_text = 0

    emb_key = "n_embd"

    datasets = ["coco", "gcc", "f30k"]
    loss_names = _loss_names({"itm": 1})
    batch_size = 512
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    val_check_interval = 0.05
    lr_mult = 10

    # Optimizer Setting
    opt_type = "adam"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    end_lr = 0

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    test_only = False

    # below params varies with the environment
    data_root = f"/project/arrows"  # path of arrow files
    log_dir = "/project/result"
    per_gpu_batchsize = 8  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32
    amp_level = "O1"

    checkpoint_dirpath = '/nas/po.ai'

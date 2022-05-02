import os
from sacred import Experiment


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


ex = Experiment("BiFrost")
ex_nmt = Experiment("ModalityTranslator")


@ex.config
def config():
    python_path = os.path.abspath('/')
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
    data_root = f"./arrows"  # path of arrow files
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


@ex_nmt.config
def config():
    # Experiment Setting
    ex_tag = ""
    seed = 0
    datasets = ["coco", "f30k"]
    data_root = f"./arrows"  # path of arrow files
    log_dir = "./result"
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 32
    amp_level = "O1"
    checkpoint_dirpath = '/nas/po.ai'
    batch_size = 512
    per_gpu_batchsize = 4
    max_epochs = 10
    max_steps = None
    loss_names = _loss_names({"itm": 1})
    val_check_interval = 0.05
    logging_interval = 10

    # Image Setting
    encoder_path = 'nf_resnet50'
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    image_only = False
    draw_false_image = 0

    # Text Setting
    lm_task = 'translate'
    lm_path = 'facebook/m2m100_418M'
    tokenizer = lm_path
    bleu_tokenizer_path = 'bert-base-uncased'
    bleu_conf = None
    vqav2_label_size = 3129
    max_text_len = 40
    whole_word_masking = False
    draw_false_text = 0
    forced_bos_token_id = None
    forced_bos_token = None
    freeze_decoder = False
    use_lm_as_encoder = False
    train_multimodality = True

    # Optimizer Setting
    opt_type = "adam"
    learning_rate = 3e-5*num_nodes*num_gpus*per_gpu_batchsize/64
    weight_decay = 0.01
    betas = (0.9, 0.95)
    nesterov = True
    end_lr = 0
    sched_type = 'cos'
    multistep_milestones = None
    multistep_decay_rate = 0.1
    sched_interval = 'epoch'
    sched_freq = 0.05
    decay_power = 1
    warmup_steps = 0.1
    lr_mult = 10

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    test_only = False

    # Vision Encoder Setting
    freeze_encoder = False
    num_frozen_stages = 0
    use_pretrained_encoder = True
    num_ds = 2

    translate_tasks = dict(i2t=1.)
    dist_backend = 'horovod'


@ex_nmt.named_config
def finetune():
    per_gpu_batchsize = 8
    opt_type = 'adam'
    sched_type = 'cos'
    freeze_encoder = True
    num_frozen_stages = 2


@ex_nmt.named_config
def bibid():
    per_gpu_batchsize = 2
    use_lm_as_encoder = True
    train_multimodality = True
    translate_tasks = dict(i2t=1., t2i=20.)


@ex_nmt.named_config
def bibid_l1():
    per_gpu_batchsize = 2
    use_lm_as_encoder = True
    train_multimodality = True
    translate_tasks = {'i2t': 1., 't2i+L1': 20.}


@ex_nmt.named_config
def bibid_full():
    per_gpu_batchsize = 2
    use_lm_as_encoder = True
    train_multimodality = True
    translate_tasks = dict(i2t=0.5, t2t=0.5, i2i=10., t2i=10.)


@ex_nmt.named_config
def bibid_l1_full():
    per_gpu_batchsize = 2
    use_lm_as_encoder = True
    train_multimodality = True
    translate_tasks = {'i2t': 0.5, 't2t': 0.5, 'i2i+L1': 10., 't2i+L1': 10.}

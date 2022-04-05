python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/bifrost_electra exp_name=electra_duel $@ --force
python /project/gen_symlinks.py
python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/bifrost_electra exp_name=electra_local $@ --force
python /project/gen_symlinks.py
python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/bifrost_electra exp_name=electra_global $@ --force
python /project/gen_symlinks.py
python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/bifrost_electra exp_name=electra_duel $@ --force
python /project/gen_symlinks.py
python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/bifrost_electra exp_name=electra_local $@ --force
python /project/gen_symlinks.py
python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/bifrost_electra exp_name=electra_global $@ --force
python /project/gen_symlinks.py
python /project/send_alarm.py
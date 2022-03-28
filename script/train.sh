if [ $FROZEN_EXP_MODE == "E0" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_small_frozen_duel $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_small_frozen_local $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_small_frozen_global $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_small_frozen_vh_duel num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_small_frozen_vh_local num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_small_frozen_vh_global num_vision_tokens=2 $@
elif [ $FROZEN_EXP_MODE == "E1" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_small_scratch_duel $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_small_scratch_local $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_small_scratch_global $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_small_scratch_vh_duel num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_small_scratch_vh_local num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_small_scratch_vh_global num_vision_tokens=2 $@
elif [ $FROZEN_EXP_MODE == "E2" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_frozen_duel $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_frozen_local $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_frozen_global $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_frozen_vh_duel num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_frozen_vh_local num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_frozen_vh_global num_vision_tokens=2 $@
elif [ $FROZEN_EXP_MODE == "E3" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_scratch_duel $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_scratch_local $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_scratch_global $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_scratch_vh_duel num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_scratch_vh_local num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_electra lm='google/electra-base-discriminator' exp_name=electra_base_scratch_vh_global num_vision_tokens=2 $@
elif [ $FROZEN_EXP_MODE == "G0" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_frozen_duel $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_frozen_local $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_frozen_global $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_frozen_vh_duel num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_frozen_vh_local num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_frozen_vh_global num_vision_tokens=2 $@
elif [ $FROZEN_EXP_MODE == "G1" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_scratch_duel $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_scratch_local $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_scratch_global $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_scratch_vh_duel num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_scratch_vh_local num_vision_tokens=10 $@
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_small_scratch_vh_global num_vision_tokens=2 $@
fi

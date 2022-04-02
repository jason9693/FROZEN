if [ $FROZEN_EXP_MODE == "EF" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_duel_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_local_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_global_vh num_vis_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "ES" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_duel_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_local_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_global_vh num_vis_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "GF" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_duel_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_local_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_global_vh num_vis_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "GS" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_duel_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_local_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_global_vh num_vis_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "BF" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_duel_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_local_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_global_vh num_vis_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "BPF" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_duel_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_local_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_global_vh num_vis_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "BS" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_scratch_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_scratch_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_scratch_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_scratch_duel_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_scratch_local_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_scratch_global_vh num_vis_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "BPS" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_duel_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_local_vh num_vis_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=16 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_global_vh num_vis_tokens=2 $@ --force
fi

if [ $FROZEN_EXP_MODE == "E0" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_frozen_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_frozen_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_frozen_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_frozen_vh_duel num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_frozen_vh_local num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_electra exp_name=electra_base_frozen_vh_global num_vision_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "E1" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_vh_duel num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_vh_local num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_electra per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_electra exp_name=electra_base_scratch_vh_global num_vision_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "G0" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_frozen_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_frozen_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_frozen_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_frozen_vh_duel num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_frozen_vh_local num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_gpt2 exp_name=gpt2_frozen_vh_global num_vision_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "G1" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_vh_duel num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_vh_local num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_gpt2 per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_gpt2 exp_name=gpt2_scratch_vh_global num_vision_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "B0" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_frozen_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_frozen_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_frozen_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_frozen_vh_duel num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_frozen_vh_local num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_bert exp_name=bert_frozen_vh_global num_vision_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "B1" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_frozen_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_frozen_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_frozen_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_frozen_vh_duel num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="local" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_frozen_vh_local num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="global" pretrained_vision=False log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_frozen_vh_global num_vision_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "B2" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_scratch_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_scratch_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_scratch_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_scratch_vh_duel num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_scratch_vh_local num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_bert exp_name=bert_scratch_vh_global num_vision_tokens=2 $@ --force
elif [ $FROZEN_EXP_MODE == "B3" ]
then
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_duel $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_local $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_global $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="duel" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_vh_duel num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="local" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_vh_local num_vision_tokens=10 $@ --force
  python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_plm per_gpu_batchsize=32 vis_mode="global" pretrained_vision=True log_dir=/project/result/frozen_bert_plm exp_name=bert_plm_scratch_vh_global num_vision_tokens=2 $@ --force
fi

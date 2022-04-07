python /project/send_alarm.py --msg "BiFrost experiment is started at session $SESSION_JOB_ID"
python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_base per_gpu_batchsize=16 vis_mode="duel" pretrained_vision=False log_dir=/project/result/bifrost_bert_base num_vis_tokens=10 $@ --force
python /project/gen_symlinks.py
python train.py with data_root=./dataset/coco/arrows num_gpus=8 num_nodes=1 task_finetune_bert_base per_gpu_batchsize=16 vis_mode="local" pretrained_vision=False log_dir=/project/result/bifrost_bert_base num_vis_tokens=8 $@ --force
python /project/gen_symlinks.py
python /project/send_alarm.py --msg "BiFrost experiment is finished at session $SESSION_JOB_ID"
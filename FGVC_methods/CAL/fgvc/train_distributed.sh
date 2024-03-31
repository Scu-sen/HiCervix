PORT=${PORT:-29500}
echo $PORT
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT train_distributed.py
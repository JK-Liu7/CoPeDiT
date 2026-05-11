export OMP_NUM_THREADS=4

torchrun --nnodes=1 --nproc_per_node=4 --master_port=29509 train_CoPeVAE_Brain.py --distributed
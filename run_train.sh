nohup python -m torch.distributed.run --nproc_per_node=2 train.py --init_lr 2e-5 >log.out 2>error.out &

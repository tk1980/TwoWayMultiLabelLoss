import torch
import random
import numpy as np

from utils.parser import get_parser
from train import Trainer


def main_worker(gpu, args):
    # Distributed training
    args.gpu_no = gpu

    if args.distributed:
        if gpu >= 0:
            print(f'Use GPU: {gpu} for training')

        # For multiprocessing dist_distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * args.ngpus_per_node + gpu

        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)

    trainer = Trainer(args=args)

    for epoch in range(args.epochs):
        trainer.train(epoch=epoch)
        trainer.validate(epoch=epoch)


def main():
    args = get_parser()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Number of GPUs
    ngpus_per_node = torch.cuda.device_count()
    print(f'device_count : {ngpus_per_node}')
    if not args.distributed:
        ngpus_per_node = 1

    # Number of the GPUs per node
    args.ngpus_per_node = ngpus_per_node

    # Number of the workers per GPU
    args.num_workers = int((args.workers + ngpus_per_node - 1) /
                              ngpus_per_node)

    # Total number of processes
    if args.distributed:
        args.world_size = args.ngpus_per_node
    else:
        args.world_size = None

    # Rank of each process
    args.rank = 0

    # Training batch size per GPU
    args.batch_size = int(args.batch_size / ngpus_per_node)

    # Validation batch size per GPU
    args.val_batch_size = int(args.batch_size / ngpus_per_node)

    if args.distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(main_worker,
                                    nprocs=args.ngpus_per_node,
                                    args=[args])
    else:
        # Simply call main_worker function
        main_worker(args.gpu, args)


if __name__ == '__main__':
    main()

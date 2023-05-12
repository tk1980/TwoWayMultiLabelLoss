import os
import torch

from .transforms import get_transforms
from .datasets import CocoDataset


def load_data_loaders(dataset, args):
    data_path = args.data

    #- transform -#
    train_transform, val_transform = get_transforms(args)

    #- dataset -#
    if dataset == 'MSCOCO':
        num_classes = 80
        instances_path_val = os.path.join(data_path, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(data_path, 'annotations/instances_train2014.json')
        data_path_val   = f'{data_path}/images/val2014' 
        data_path_train = f'{data_path}/images/train2014'
        val_dataset = CocoDataset(data_path_val,
                                    instances_path_val, val_transform)
        train_dataset = CocoDataset(data_path_train,
                                    instances_path_train, train_transform)

    #- loader -#
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset) if args.distributed else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        sampler=val_sampler
        )

    print(f"Training data size: {len(train_loader.dataset)}")
    print(f"Validation data size: {len(val_loader.dataset)}")

    train_len = ((len(train_loader.dataset) - 1) // (args.batch_size * args.ngpus_per_node)) + 1
    val_len = ((len(val_loader.dataset) - 1) // (args.val_batch_size * args.ngpus_per_node)) + 1
    return train_loader, val_loader, num_classes, train_len, val_len
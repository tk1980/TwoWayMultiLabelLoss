import os
import time
import builtins
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from utils.data_loader import load_data_loaders
from utils.criterion import get_criterion
from utils.optimizers import adjust_learning_rate
from utils.optimizers import load_optimizer
from utils.utils import AverageMeterCollection
from utils.utils import compute_performance
from utils.utils import reduce_tensor

from torch.utils.tensorboard import SummaryWriter

from torch.cuda.amp import GradScaler, autocast

# Training class
class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Mute the other workers 
        if not self.is_primary_worker():
            def print_pass(*args):
                pass
            builtins.print = print_pass

        # Set training and validation data loaders
        self.train_loader, self.val_loader, \
            self.args.num_classes, self.args.train_len, self.args.val_len = \
            load_data_loaders(dataset=self.args.dataset,
                              args=self.args)

        # Set backbone network
        if hasattr(torchvision.models, args.arch):
            if args.finetune:
                self.model = getattr(torchvision.models, args.arch)(pretrained=True)
                # Replace FC layer
                for fc_name, fc_layer in self.model.named_modules():
                    if type(fc_layer) is nn.Linear:
                        break
                setattr(self.model, fc_name, nn.Linear(fc_layer.in_features, args.num_classes))
            else:
                self.model = getattr(torchvision.models, args.arch)(num_classes=args.num_classes)
        else:
            raise ValueError(
                f"Not supported model architecture {args.arch}")
        
        if self.is_primary_worker():
            print(self.model)
            self.writer = SummaryWriter(log_dir=os.path.join(args.output, 'logs'))

        # Set criterion and opimizer
        self.criterion = get_criterion(self.args)

        self.optimizer = load_optimizer(self.args, self.model)
        if self.is_primary_worker():
            print(self.optimizer)

        # Distributed data parallel
        if self.args.distributed:
            torch.cuda.set_device(device=self.args.gpu_no)
            self.model.cuda(device=self.args.gpu_no)
            self.model = torch.nn.parallel.DistributedDataParallel(
                module=self.model,
                device_ids=[self.args.gpu_no])
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.criterion.cuda(device=self.args.gpu_no)
        
        torch.backends.cudnn.benchmark = True

    # Training on one mini-batch
    def train_minibatch(self, iteration, batch, epoch, batch_start_time,
                    scaler, meters):
        image, target = batch

        batch_size = image.size(0)

        current_lr = adjust_learning_rate(
            optimizer=self.optimizer,
            epoch=epoch,
            iteration=iteration,
            lr_decay_type=self.args.lr_decay_type,
            epochs=self.args.epochs,
            train_len=self.args.train_len)

        target = target.cuda()
        image = image.cuda()

        # forward
        with autocast():  # mixed precision
            output = self.model(image).float()

        # loss
        loss = self.criterion(output, target)

        # backward
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        self.optimizer.zero_grad()

        # print intermediate results
        if (iteration + 1) % self.args.print_freq == 0:
            meters = compute_performance(output=output,
                                        target=target,
                                        meters=meters,
                                        world_size=self.args.world_size)
            reduced_loss = reduce_tensor(loss.data, 
                                        self.args.world_size)
            if self.args.distributed:
                torch.cuda.synchronize()
            meters.get('losses').update(reduced_loss.item(), batch_size)
        meters.get('batch_time').update( (time.time() - batch_start_time) )

        return meters, current_lr

    
    # Training on one epoch
    def train(self, epoch):
        meters = AverageMeterCollection('batch_time', 'losses', 'mapcls', 'mapsmp')

        # Init trainer 
        self.model.train()
        self.optimizer.zero_grad()
        scaler = GradScaler()

        # Init timer 
        tic = time.time()
        batch_start_time = time.time()
        
        if self.is_primary_worker():
            tqdm_batch = tqdm(total=len(self.train_loader), desc="[Epoch {}]".format(epoch))

        # Train over whole mini-batches
        for iteration, batch in enumerate(self.train_loader):
            meters, lr = self.train_minibatch(batch=batch, iteration=iteration, epoch=epoch,
                             batch_start_time=batch_start_time,
                             scaler=scaler, meters=meters)
            batch_start_time = time.time()

            if self.is_primary_worker():
                tqdm_batch.set_postfix({'Time': '{batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                                                                                            batch_time=meters.batch_time),
                                        'Loss': '{loss.val:.3f} ({loss.avg:.3f})'.format(loss=meters.losses), 
                                        'mAP@cls': '{map.val:.2f} ({map.avg:.2f})'.format(map=meters.mapcls), 
                                        'mAP@sample': '{map.val:.2f} ({map.avg:.2f})'.format(map=meters.mapsmp) })
                tqdm_batch.update()

        if self.is_primary_worker():
            tqdm_batch.close()
            time_spent = time.time() - tic
            print('[Epoch {}] {:.3f} sec/epoch'.format(epoch, time_spent), end='\t')
            print('remaining time: {:.3f} hours'.format( (self.args.epochs - epoch - 1) * time_spent / 3600))
            self.writer.add_scalar('LearningRate', lr, epoch)
            self.writer.add_scalar('Loss/train', meters.losses.avg, epoch)
            self.writer.add_scalar('mAP_class/train', meters.mapcls.avg, epoch)
            self.writer.add_scalar('mAP_sample/train', meters.mapsmp.avg, epoch)
            self.writer.close()


    # Validation
    def validate(self, epoch=0):
        meters = AverageMeterCollection('mapcls', 'mapsmp')

        # Init 
        self.model.eval()

        # Validate over whole mini-batches
        output, target = [], []
        for iteration, (image, target_) in enumerate(self.val_loader):
            image = image.cuda()
            target_ = target_.cuda()

            # forward pass and compute loss
            with torch.no_grad():
                output_ = self.model(image)
                output.append(output_)
                target.append(target_)

        output = torch.cat(output)
        target = torch.cat(target)
        meters = compute_performance(output=output,
                                    target=target,
                                    meters=meters,
                                    world_size = self.args.world_size)

        if self.is_primary_worker():
            print('Test: mAP@cls {mapcls.avg:.3f}, mAP@sample {mapsmp.avg:.3f}'
                .format(mapcls=meters.mapcls, mapsmp=meters.mapsmp))

            self.writer.add_scalar('mAP_class/test', meters.mapcls.avg, epoch)
            self.writer.add_scalar('mAP_sample/test', meters.mapsmp.avg, epoch)
            self.writer.close()
            
            self.save_checkpoint(epoch=epoch, meters=meters)

    
    # Check primary worker or not
    def is_primary_worker(self):
        return not self.args.distributed or \
               (self.args.distributed and
                (self.args.rank % self.args.ngpus_per_node) == 0)

    # Load checkpoint
    def load_checkpoint(self, weight_file):
        if os.path.isfile(weight_file):
            print(f"=> loading checkpoint '{weight_file}'")
            checkpoint = torch.load(weight_file)

            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            self.model.load_state_dict(checkpoint)
            print(f"=> checkpoint loaded '{weight_file}'")
        else:
            raise Exception(f"=> no checkpoint found at '{weight_file}'")

    # Save checkpoint
    def save_checkpoint(self, epoch, meters):
        if self.is_primary_worker():
            save_dict = {
                'epoch': epoch + 1,
                'arch': self.args.arch,
                'state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') 
                                else self.model.state_dict(),
                'mAPclass': meters.mapcls.avg,
                'mAPsample': meters.mapsmp.avg,
                'optimizer': self.optimizer.state_dict(),
            }

            checkpoint_dir = os.path.join(self.args.output, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            filepath = f"{checkpoint_dir}/checkpoint-{self.args.arch}-last.pth"
            torch.save(save_dict, filepath)
    
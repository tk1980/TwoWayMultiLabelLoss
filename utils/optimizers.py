import torch.optim as optim
import numpy as np

# Set optimizer
def load_optimizer(args, model):
    if args.finetune:
        param_group = add_learning_rate(model, args.lr, 
                                        backbone_lr=args.finetune_lr, 
                                        finetune_layer=args.finetune_layer)
    else:
        param_group = model.parameters()

    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            param_group,
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    else:
        raise ValueError("Unknown optimizer : {}".format(args.optimizer))

    set_init_lr(optimizer.param_groups)

    return optimizer


# Initial learning rate
def set_init_lr(param_groups):
    for group in param_groups:
        group['init_lr'] = group['lr']


# Touch learning rates on finetuned layers
def add_learning_rate(model, init_lr, backbone_lr=0.1, finetune_layer='fc'):
    finetune_layer += '.'
    backbone = []
    finetune = []
    print(f'Finetuning layers lr: {init_lr}:')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if finetune_layer in name:
            finetune.append(param)
            print(name)
        else:
            backbone.append(param)
    print('The other layers lr: {}:'.format(init_lr * backbone_lr))
    return [
        {'params': finetune, 'lr': init_lr},
        {'params': backbone, 'lr': init_lr * backbone_lr}]


# Cosine learning rate
def adjust_learning_rate_cosine(epoch, iteration, dataset_len,
                                epochs):
    total_iter = epochs * dataset_len
    current_iter = iteration + epoch * dataset_len

    lr = 1 / 2 * (np.cos(np.pi * current_iter / total_iter) + 1)

    return lr


# Set learning rate
def adjust_learning_rate(optimizer, epoch, iteration, lr_decay_type,
                         epochs, train_len):
    if lr_decay_type == 'cos':
        lr_factor = adjust_learning_rate_cosine(
                                epoch=epoch,
                                iteration=iteration,
                                dataset_len=train_len,
                                epochs=epochs)
    else:
        raise ValueError("Unknown lr decay type {}.".format(lr_decay_type))


    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['init_lr'] * lr_factor

    return optimizer.param_groups[0]['lr']

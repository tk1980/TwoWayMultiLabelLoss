import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Training')
    # data
    parser.add_argument('data', help='path to dataset')
    parser.add_argument('--dataset', default='MSCOCO', type=str)

    # model
    parser.add_argument('--arch', default='resnet50')
    
    # optimization
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr-decay-type', default='cos', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=True, type=bool)

    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--workers', default=24, type=int)
  
    # finetune 
    parser.add_argument('--finetune', default=True, type=bool)
    parser.add_argument('--finetune-lr', default=0.1, type=float)
    parser.add_argument('--finetune-layer', default='fc', type=str)

    # data augmentation 
    parser.add_argument('--image-size', default=224, type=int)
    parser.add_argument('--crop-size', default=256, type=float)

    # loss
    parser.add_argument('--loss', default='TwoWayLoss', type=str)
    parser.add_argument('--loss-Tp', default=4., type=float)
    parser.add_argument('--loss-Tn', default=1., type=float)

    # misc
    parser.add_argument('--output', default='./result', type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--print-freq', default=10, type=int)
    
    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:6946', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int)
    
    args =  parser.parse_args()

    return args
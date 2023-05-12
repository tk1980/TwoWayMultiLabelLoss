import torch
import numpy as np

# Average precision
def AP(output, target):
    if len(target) == 0 or np.all(target==0):
        return -1

    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ > 0
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

# mean Average precision
def mAP(targs, preds):
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        nzidx = targets >= 0
        # compute average precision
        ap[k] = AP(scores[nzidx], targets[nzidx])
    return 100 * ap[ap>=0].mean()    

# Compute performances of mAP at class and sample
def compute_performance(output, target, meters, world_size=None):
    if world_size is not None:
        output_all = concat_distributed_tensors(
            tensor=output.data,
            world_size=world_size).detach()
        target_all = concat_distributed_tensors(
            tensor=target,
            world_size=world_size).detach()
    else:
        output_all, target_all = output.data, target.data

    mAP_class = mAP(target_all.cpu().numpy(), output_all.cpu().numpy())
    mAP_sample = mAP(target_all.t().cpu().numpy(), output_all.t().cpu().numpy())

    # mAP at class    
    meters.mapcls.update(mAP_class, output_all.size(0))
    # mAP at sample    
    meters.mapsmp.update(mAP_sample, output_all.size(0))

    return meters

# Process tensors across workers
def concat_distributed_tensors(tensor, world_size):
    rt = tensor.clone()

    size_ten = torch.IntTensor([rt.shape[0]]).to(rt.device)
    gather_size = [torch.ones_like(size_ten) for _ in range(world_size)]
    torch.distributed.all_gather(tensor=size_ten, tensor_list=gather_size)
    max_size = torch.cat(gather_size, dim=0).max()

    padded = torch.empty(max_size, *rt.shape[1:],
                         dtype=rt.dtype,
                         device=rt.device)
    padded[:rt.shape[0]] = rt
    gather_t = [torch.ones_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(tensor=padded, tensor_list=gather_t)

    slices = []
    for i, sz in enumerate(gather_size):
        slices.append(gather_t[i][:sz.item()])

    concat_t = torch.cat(slices, dim=0)
    
    return concat_t

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    if world_size is not None:
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= world_size
    return rt

# Performance meters
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterCollection(object):
    def __init__(self, *meter_names):
        for name in meter_names:
            setattr(self, name, AverageMeter())

    def get(self, meter_name):
        return getattr(self, meter_name)

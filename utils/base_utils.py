import time
import torch
from torch.autograd import Variable


def to_var(x, requires_grad=True):
    # 转成 torch.autograd.Variable
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def to_numpy(x):
    assert isinstance(x, torch.Tensor)
    return x.detach().cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def get_curtime():
    current_time = time.strftime('%b%d_%H%M%S', time.localtime())
    return current_time

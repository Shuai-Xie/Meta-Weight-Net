import torch
import time

from utils import AverageMeter, to_var


def adjust_learning_rate(lr, optimizer, epoch):
    # divided by 10 after 80 and 90 epoch (for a total 100 epochs)
    # 这种写法很简洁
    lr = lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    # lr = lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 70)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))  # acc%
    return res


@torch.no_grad()
def validate(val_loader, model, criterion,
             epoch, print_freq, writer):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    begin_step = (epoch - 1) * len(val_loader)

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var.data, topk=(1,))[0]

        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

        writer.add_scalar('Test/loss', losses.avg, global_step=begin_step + i)

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    writer.add_scalar('Test/top1_acc', top1.avg, global_step=epoch)

    return top1.avg

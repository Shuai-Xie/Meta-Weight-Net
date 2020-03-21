import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import random
import copy
import numpy as np
from pprint import pprint
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.data_utils import get_imb_meta_test_datasets
from net.resnet import build_model
from net.vnet import VNet
from engine import *
from utils import get_curtime, save_model
from plt_vnet import get_vnet_map

# parse arguments, args 默认会把参数名中 '-' 转成 '_'
parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',  # train bs
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=10)  # cifar10
parser.add_argument('--num_meta', type=int, default=10,  # meta data, 10 samples/class
                    help='The number of meta data for each class.')
parser.add_argument('--imb_factor', type=int, default=50)  # imbalance factor
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',  # test bs todo: bs 不同会影响测试 acc 吗
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',  # total 100 epoches
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,  # init lr=0.1
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')  # momentum=0.9
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,  # decay
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42, metavar='S',  # random
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--tag', default='exp', type=str,
                    help='experiment tag to create tensorboard, model save dir name')

params = [
    '--dataset', 'cifar100',
    '--num_classes', '100',
    '--imb_factor', '50',
    '--tag', 'mw_v1'
]
args = parser.parse_args(params)
pprint(vars(args))

use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True

# random.shuffle 重现
# torch.manual_seed(args.seed)  # not work!!!
random.seed(args.seed)
imb_train_dataset, meta_train_dataset, test_dataset = get_imb_meta_test_datasets(
    args.dataset, args.num_classes, args.num_meta, args.imb_factor
)

# imb_train/valid_meta/test
kwargs = {'num_workers': 4, 'pin_memory': True}
imb_train_loader = DataLoader(imb_train_dataset,
                              batch_size=args.batch_size,
                              drop_last=True,
                              shuffle=True, **kwargs)
valid_loader = DataLoader(meta_train_dataset,  # 总共 10*10=100
                          batch_size=args.batch_size,  # 100
                          drop_last=True,
                          shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False, **kwargs)
print('load imb dataset done!')

"""
Meta-Weight-Net v1
- features before softmax as vnet input. [logits]
- VNet(num_classes, 100, 1)
"""

if __name__ == '__main__':
    # 定义 2个 model
    # classifier: meta ResNet32
    model = build_model(args.dataset).cuda()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,  # lr 阶段性变化
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    # v1: logits as vnet input
    vnet = VNet(args.num_classes, 100, 1).cuda()
    optimizer_c = torch.optim.SGD(vnet.params(), 1e-5,  # lr 不变
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    print('build model done!')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    exp = f'{args.tag}_{args.dataset}_imb{args.imb_factor}_{get_curtime()}'
    print('exp:', exp)

    writer = SummaryWriter(log_dir=os.path.join('runs', exp))
    model_save_dir = os.path.join('output', exp)
    os.makedirs(model_save_dir, exist_ok=True)

    best_prec1, best_epoch = 0, 0
    curves_data = []

    for epoch in range(1, args.epochs + 1):
        # 调整 classifier optimizer 的 lr = meta_lr
        adjust_learning_rate(args.lr, optimizer_a, epoch)

        # meta train on (imb_train_data, meta_data)
        train_mw_v1(imb_train_loader, valid_loader,
                    model, vnet,
                    args.lr,
                    optimizer_a, optimizer_c,
                    epoch, args.print_freq, writer)

        # save vnet loss-weight map each epoch
        # with torch.no_grad():
        #     x, y = get_vnet_map(vnet)
        #     curves_data.append((x, y))
        #     curves_data_npy = np.array(curves_data).squeeze()
        #     np.save(f'npy/{exp}_curves_data.npy', curves_data_npy)  # (100, 2, 100)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion,
                         epoch, args.print_freq, writer)

        # remember best prec@1 and save checkpoint
        if prec1 > best_prec1:
            best_prec1, best_epoch = prec1, epoch
            save_model(os.path.join(model_save_dir, 'rs32_epoch_{}.pth'.format(epoch)),
                       model, epoch, best_prec1)
            save_model(os.path.join(model_save_dir, 'vnet_epoch_{}.pth'.format(epoch)),
                       vnet, epoch, best_prec1)

    print('Best accuracy: {}, epoch: {}'.format(best_prec1, best_epoch))

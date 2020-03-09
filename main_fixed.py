import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import random
import copy
import numpy as np
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.data_utils import build_dataset, get_cls_img_idxs_dict, get_img_num_per_cls
from net.resnet import build_model
from net.vnet import load_vnet
from engine import adjust_learning_rate, validate, train_mw_fixed
from utils import get_curtime, save_model

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
    '--dataset', 'cifar10',
    '--num_classes', '10',
    '--imb_factor', '10',
    '--tag', 'mw_fixed'
]
args = parser.parse_args(params)
pprint(vars(args))

kwargs = {'num_workers': 4, 'pin_memory': True}
use_cuda = not args.no_cuda and torch.cuda.is_available()

# 增加复现性，因为 build_dataset 内用到 np.random.shuffle
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True

# same to torchvision.datasets.CIFAR10 class, but data has been split by num_meta
meta_train_dataset, train_dataset, test_dataset = build_dataset(args.dataset, args.num_meta)

# make imbalanced data
torch.manual_seed(args.seed)
classe_labels = range(args.num_classes)

# 每个 cls 对应的 img idxs list, dict{cls_idx : img_id_list, ..}
data_list = get_cls_img_idxs_dict(train_dataset.targets, args.num_classes)

# 指数不平衡 imb_factor = 10
img_num_list = get_img_num_per_cls(args.dataset, args.imb_factor, args.num_meta * args.num_classes)
# [5000, 3871, 2997, 2320, 1796, 1391, 1077, 834, 645, 500]

idx_to_del = []  # 存储各类 要 del 的 img idxs
for cls_idx, img_id_list in data_list.items():
    random.shuffle(img_id_list)
    img_num = img_num_list[int(cls_idx)]
    idx_to_del.extend(img_id_list[img_num:])  # 除去 imb 指定的，剩下的全部删去

# build imbalance dataset
imb_train_dataset = copy.deepcopy(train_dataset)
# 删除各类取 imb_train_dataset 后剩下的
imb_train_dataset.data = np.delete(train_dataset.data, idx_to_del, axis=0)
imb_train_dataset.targets = np.delete(train_dataset.targets, idx_to_del, axis=0)

# imb_train/valid_meta/test
imbalanced_train_loader = DataLoader(imb_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
validation_loader = DataLoader(meta_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
print('load imb dataset done!')

if __name__ == '__main__':
    # 定义 2个 model
    # classifier: meta ResNet32
    model = build_model(args.dataset).cuda()
    optimizer_a = torch.optim.SGD(model.params(), args.lr,  # lr 阶段性变化
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    # fixed vnet from pretrain
    vnet = load_vnet(ckpt_path='output/mw_cifar10_imb10_Mar05_16-54-26/vnet_epoch_94.pth')
    vnet.eval()
    print('build model done!')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    dirname = f'{args.tag}_{args.dataset}_imb{args.imb_factor}_{get_curtime()}'
    print('exp:', dirname)
    # mw_cifar10_imb10_Mar05_23-03-29

    writer = SummaryWriter(log_dir=os.path.join('runs', dirname))
    model_save_dir = os.path.join('output', dirname)
    os.makedirs(model_save_dir, exist_ok=True)

    best_prec1, best_epoch = 0, 0

    for epoch in range(1, args.epochs + 1):
        # 调整 classifier optimizer 的 lr = meta_lr
        adjust_learning_rate(args.lr, optimizer_a, epoch)

        # meta train on (imb_train_data, meta_data)
        train_mw_fixed(imbalanced_train_loader, model, vnet,
                       optimizer_a,
                       epoch, args.print_freq, writer)

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
    # imb10,

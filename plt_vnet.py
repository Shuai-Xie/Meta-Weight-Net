import torch
from net.vnet import VNet
from net.resnet import build_model
from utils import to_numpy, load_model, to_var
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

"""
plot loss-weight curve with trained vnet
"""

save_dir = 'npy'
plot_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


def cvt_and_save_np(x, y, xy_path=None):
    x, y = to_numpy(x).squeeze(), to_numpy(y).squeeze()
    if xy_path is not None:
        np.save(xy_path, (x, y))
    return x, y


@torch.no_grad()
def get_vnet_map(vnet, upper_loss=10, num_pts=100, xy_path=None):
    # x: loss, y: weight
    x = torch.linspace(0, upper_loss, steps=num_pts).reshape((num_pts, 1)).cuda()
    y = vnet(x)
    x, y = cvt_and_save_np(x, y, xy_path)
    return x, y


@torch.no_grad()
def get_vnet_v2_map(vnet, upper_loss=10, num_pts=100, xy_path=None):
    # batch_losses -> batch_weights
    # x: loss, y: weight
    x = torch.linspace(0, upper_loss, steps=num_pts).reshape((1, num_pts)).cuda()  # batch_size 不同
    y = vnet(x)
    x, y = cvt_and_save_np(x, y, xy_path)
    return x, y


def plt_xy_map(x, y, title):
    # loss-weight curve
    plt.title(title)
    plt.xlabel('loss')
    plt.ylabel('weight')
    plt.ylim(0.5, 1.0)
    plt.plot(x, y)
    plt.savefig('{}/{}.png'.format(plot_dir, title), bbox_inches='tight')
    plt.show()


def plt_vnet(exp='mw_cifar10_imb10'):
    upper_loss, num_pts = 20, 100
    best_epoch = 94

    # get x,y
    xy_path = '{}/{}_xy_loss{}.npy'.format(save_dir, exp, upper_loss)
    if os.path.exists(xy_path):
        print('load datapoints from npy')
        x, y = np.load(xy_path)
    else:
        print('load datapoints from model')
        vnet = VNet(1, 100, 1).cuda()
        vnet = load_model(vnet, ckpt_path=f'output/{exp}_Mar05_165426/vnet_epoch_{best_epoch}.pth')
        x, y = get_vnet_map(vnet, upper_loss, num_pts, xy_path)

    # plot map
    plt_xy_map(x, y, title='{}_loss{}'.format(exp, upper_loss))


if __name__ == '__main__':
    # plt_vnet()
    vnet = VNet(100, 300, 100).cuda()
    vnet = load_model(vnet, ckpt_path='output/mw_v2_cifar10_imb100_Mar10_132504/vnet_epoch_89.pth')
    vnet.eval()

    model = build_model('cifar10').cuda()
    model = load_model(model, ckpt_path='output/mw_v2_cifar10_imb100_Mar10_132504/rs32_epoch_89.pth')
    model.eval()

    print('load model done!')

    from main_mw_v2 import imbalanced_train_loader

    # 在 meta_model 第2步更新时 选择 high info 样本?
    # 2次 asm?

    for i, (input, target) in enumerate(imbalanced_train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        with torch.no_grad():
            y_f = model(input_var)  # [100,10]
            cost_w = F.cross_entropy(y_f, target_var, reduction='none')
            cost_v = torch.reshape(cost_w, (1, len(cost_w)))  # [1,100] batchsize=1 for vnet

            w_new = vnet(cost_v)  # [1,100]

            x, y = to_numpy(cost_v), to_numpy(w_new)
            x, y = x.T, y.T
            plt.xlabel('loss')
            plt.ylabel('weight')
            plt.scatter(x, y)
            plt.show()

            if i == 5:
                break

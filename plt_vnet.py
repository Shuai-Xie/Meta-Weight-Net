import torch
from net.vnet import load_vnet
from utils import to_numpy
import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = 'npy'
plot_dir = 'plots'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)


def get_loss_weight_map(vnet, upper_loss=10, num_pts=100, xy_path=None):
    # x: loss, y: weight
    x = torch.linspace(0, upper_loss, steps=num_pts).reshape((num_pts, 1)).cuda()
    y = vnet(x)
    x, y = to_numpy(x), to_numpy(y)
    # save npy
    if xy_path is not None:
        np.save(xy_path, (x, y))
    return x, y


def plt_map(x, y, title):
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
        vnet = load_vnet(ckpt_path=f'output/{exp}_Mar05_165426/vnet_epoch_{best_epoch}.pth')
        x, y = get_loss_weight_map(vnet, upper_loss, num_pts, xy_path)

    # plot map
    plt_map(x, y, title='{}_loss{}'.format(exp, upper_loss))


if __name__ == '__main__':
    plt_vnet()

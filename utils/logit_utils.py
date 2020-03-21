import numpy as np
import torch.nn.functional as F
import torch
import random


# https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
# a = np.random.dirichlet(np.ones(10), size=1)

def generate_sum1_list(max_p, length=10, divide=2):
    sum1_list = []
    half_p = 1 - max_p  # 剩下的
    for i in range(length - 1):
        half_p /= divide
        if sum(sum1_list) + half_p <= 1 - max_p:
            sum1_list.append(half_p)
        if len(sum1_list) == length - 1:
            break
        if half_p < 1e-3:
            remain_num = length - 1 - len(sum1_list)
            remain_p = 1 - max_p - sum(sum1_list)
            sum1_list.extend([remain_p / remain_num] * remain_num)
            break
    # random.shuffle(sum1_list)  # 后面 pro 可随意
    sum1_list = [max_p] + sum1_list
    # print(max_p, sum1_list)
    return sum1_list


def cal_entropy(y_pred):
    # log(p) 表示信息量，entropy 表示所有信息量的期望
    # https://blog.csdn.net/tsyccnh/article/details/79163834
    entropy = -np.nansum(np.multiply(y_pred, np.log(y_pred)))  # treate nan as 0, 如 log(0)
    return entropy


def see_ce_loss():
    num_pts = 100
    init_p = 0
    step = (1 - init_p) / num_pts
    for i in range(num_pts):
        max_p = init_p + i * step
        a = np.array(generate_sum1_list(max_p, length=100)).reshape(1, -1)
        a = torch.from_numpy(a)
        loss = F.cross_entropy(a, torch.tensor([0])).item()
        print(i, max_p, loss)


def manual_probs():
    p = 0.
    a = [p] + [(1 - p) / 9] * 9
    a = np.array(a).reshape(1, -1)
    a = torch.from_numpy(a)
    loss = F.cross_entropy(a, torch.tensor([0])).item()
    print(loss)


see_ce_loss()
# manual_probs()

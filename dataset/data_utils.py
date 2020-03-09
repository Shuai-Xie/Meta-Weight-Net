import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
import copy


def get_cls_img_idxs_dict(ds_labels, num_classes):
    """
    parse ds_labels into each cls img idxs
    @param ds_labels: dataset labels
    @param num_classes:
    @return: dict{ cls_idx : img_id_list,.. }
    """
    data_list_val = {}
    for j in range(num_classes):
        data_list_val[j] = [i for i, label in enumerate(ds_labels) if label == j]
    return data_list_val


def build_dataset(dataset, num_meta):
    """
    @param dataset: dataset name
    @param num_meta: meta data num per class
    @return:
        train_data_meta, train_data, test_dataset # same to torchvision.datasets.CIFAR10 class
    """
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        # pad, 4D input tensor, lrtb [可处理 1D,2D,3D 输入，分别对应不同 pad 输入]
        # same to np.pad, 反射 32x32 -> 40x40
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                          (4, 4, 4, 4), mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),  # 32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if dataset == 'cifar10':  # 5000/1000 * 10
        train_dataset = torchvision.datasets.CIFAR10(root='/nfs/xs/Datasets/CIFAR10', train=True, download=True,
                                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('/nfs/xs/Datasets/CIFAR10', train=False,
                                                    transform=transform_test)
        img_num_list = [num_meta] * 10  # 10 cls, each cls has num_meta samples
        num_classes = 10

    elif dataset == 'cifar100':  # 500/100 * 100
        train_dataset = torchvision.datasets.CIFAR100(root='/nfs/xs/Datasets/CIFAR100', train=True, download=True,
                                                      transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100('/nfs/xs/Datasets/CIFAR100', train=False,
                                                     transform=transform_test)
        img_num_list = [num_meta] * 100
        num_classes = 100
    else:
        raise ValueError('no such dataset!')

    data_list_val = get_cls_img_idxs_dict(train_dataset.targets, num_classes)

    # store train/meta img idxs, for np.delete
    idx_to_meta = []
    idx_to_train = []
    for cls_idx, img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]  # 10, num_meta
        idx_to_meta.extend(img_id_list[:img_num])  # top 10 as meta
        idx_to_train.extend(img_id_list[img_num:])  # remain as train

    # deep copy, use 2 times
    train_data_meta = copy.deepcopy(train_dataset)
    train_data = copy.deepcopy(train_dataset)

    # np delete train img idxs
    train_data_meta.data = np.delete(train_data_meta.data, idx_to_train, axis=0)
    train_data_meta.targets = np.delete(train_data_meta.targets, idx_to_train, axis=0)

    # np delete meta img idxs
    train_data.data = np.delete(train_data.data, idx_to_meta, axis=0)
    train_data.targets = np.delete(train_data.targets, idx_to_meta, axis=0)

    return train_data_meta, train_data, test_dataset


def get_img_num_per_cls(dataset, imb_factor=None, num_meta=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution 数量服从指数分布
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    if num_meta is None:
        num_meta = 0

    if dataset == 'cifar10':
        img_max = (50000 - num_meta) / 10
        cls_num = 10
    elif dataset == 'cifar100':
        img_max = (50000 - num_meta) / 100
        cls_num = 100
    else:
        return ValueError('no such dataset!')

    if imb_factor is None:  # no imbalance, return base_num = img_max
        return [img_max] * cls_num

    # process imbalance
    img_num_per_cls = []
    imb_factor = 1 / imb_factor  # 从 img_max 开始乘
    for cls_idx in range(cls_num):
        # from class 0-9
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls


if __name__ == '__main__':
    imb_num = get_img_num_per_cls('cifar10', imb_factor=100, num_meta=0)
    print(imb_num)
    print(sum(imb_num))
    """
    imb10
        [5000, 3871, 2997, 2320, 1796, 1391, 1077, 834, 645, 500]
        20431, 204 batches
    
    imb100
        [5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
        12406
    """

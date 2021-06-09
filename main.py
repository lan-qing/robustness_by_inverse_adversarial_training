import os
import random
import functools

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse
import torchattacks

from utils import *
from model import *
from attacks import *
from visualization import to_img


parser = argparse.ArgumentParser(description='Inverse adversarial training')
parser.add_argument('--normal', help='not use inverse training', type=bool, default=False)
args = parser.parse_args()
if args.normal:
    print("Normal!")

if __name__ == '__main__':
    seed = 25
    print("Use random seed ", seed)
    signature = "test"
    rootpath = f"results/{signature}_seed{seed}/"
    if not os.path.isdir(rootpath):
        os.mkdir(rootpath)

    set_seed(seed)

    NTrainPointsMNIST = 60000
    batch_size = 100
    log_interval = 1

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR10(root='/slstore/tianfeng/data', train=True, download=True, transform=transform_train)
    valset = datasets.CIFAR10(root='/slstore/tianfeng/data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                            num_workers=3)
    net = ResNet20Wrapper()
    lr = 0.1
    for epoch in range(200):
        if epoch in [100, 150]:
            lr /= 10
        if args.normal:
            net.fit(trainloader, lr=lr, epoch=epoch, adv=torchattacks.PGD(net.model, eps=4/255, alpha=2/255, steps=4))
        else:
            net.fit_inverse(trainloader, lr=lr, epoch=epoch, adv=torchattacks.PGD(net.model, eps=4/255, alpha=2/255, steps=4))
        net.validate(valloader)
        net.validate(valloader, adv=torchattacks.PGD(net.model, eps=4/255, alpha=2/255, steps=4))
    net.save(rootpath + "resnet_200_inverse.pt")

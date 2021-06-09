import numpy as np
import torch
import torch.nn as nn
import torchattacks

from utils import *

from models.lenet import Lenet
from visualization import to_img
from models.resnet import resnet20



class NetWrapper():
    def __init__(self):
        cprint('c', '\nNet:')
        self.model = None

    def fit(self, train_loader):
        raise NotImplementedError

    def predict(self, test_loader):
        raise NotImplementedError

    def validate(self, val_loader):
        raise NotImplementedError

    def save(self, filename='checkpoint.pt'):
        state = {
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state['state_dict'])


class LenetWrapper(NetWrapper):
    def __init__(self, half=False, cuda=True, double=False):
        super(LenetWrapper).__init__()
        self.model = Lenet()
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.01, weight_decay=0.0, epoch=None, adv=None):
        optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        criterion = nn.CrossEntropyLoss().cuda()
        loss, prec = train(train_loader, self.model, criterion, optimizer, epoch, half=self.half, double=self.double,
                           adv=adv)

        return loss, prec

    def fit_inverse(self, train_loader, lr=0.01, weight_decay=0.0, epoch=None, adv=None):
        pass

    def validate(self, val_loader, adv=None):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, adv=adv, half=self.half, double=self.double)
        return loss, prec


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


class ResNet20Wrapper(NetWrapper):
    def __init__(self, half=False, cuda=True, double=False):
        super(ResNet20Wrapper).__init__()
        self.resnet = resnet20()
        self.norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = nn.Sequential(self.norm_layer,
                                   self.resnet
                                   )
        self.half = half
        self.double = double
        if self.half:
            self.model.half()
        if self.double:
            self.model.double()
        if cuda:
            self.model.cuda()

    def fit(self, train_loader, lr=0.1, weight_decay=1e-4, epoch=None, adv=None):
        optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        if self.double:
            criterion.double()

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss, prec = train(train_loader, self.model, criterion, optimizer, epoch, half=self.half, double=self.double,
                           adv=adv)

        return loss, prec

    def fit_inverse(self, train_loader, lr=0.1, weight_decay=1e-4, epoch=None, adv=None):
        optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum=0.9, weight_decay=weight_decay)

        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        attack = torchattacks.PGD(self.model, eps=16/255, alpha=4/255, steps=4)
        attack.set_mode_targeted()
        loss, prec = train_inverse(train_loader, self.model, attack, optimizer, epoch, half=self.half, double=self.double,
                           adv=adv)

        return loss, prec

    def predict(self, test_loader):
        pass

    def validate(self, val_loader, adv=None):
        criterion = nn.CrossEntropyLoss().cuda()
        if self.half:
            criterion.half()
        loss, prec = validate(val_loader, self.model, criterion, half=self.half, double=self.double, adv=adv)
        return loss, prec

import torch
import torch.nn as nn
import numpy as np


def pgd_attack_reverse(model, images, labels, eps=1.0, alpha=0.1, iters=10, half=False, double=False):
    images = images.cuda()
    labels = labels.cuda()
    loss = nn.CrossEntropyLoss()
    if half:
        loss.half()
    if double:
        loss.double()
    ori_images = images.data
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        # if images.grad.max() < 1e-10:
        #     print(i)
        #     print(cost)
        #     print(images)
        #     print(outputs.argmax(dim=1))
        #     print(outputs)
        #     print("===============")
        #     print(model(images[0], verbose=True))
        #     print(model(images[1], verbose=True))
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images


def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=40, half=True, double=False):
    images = images.cuda()
    labels = labels.cuda()
    loss = nn.CrossEntropyLoss()
    if half:
        loss.half()
    if double:
        loss.double()
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels)  # .to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

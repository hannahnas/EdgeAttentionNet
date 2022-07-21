import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def cross_entropy_loss2d(inputs, targets, cuda=False, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()

    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid

    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    weights = Variable(weights)
    inputs = torch.sigmoid(inputs)
    loss = nn.BCELoss(weights, reduction='mean')(inputs, targets)

    return loss


def re_Dice_Loss(inputs, targets, cuda=False, balance=1.1):
    n, c, h, w = inputs.size()
    smooth=1
    inputs = torch.sigmoid(inputs)  # F.sigmoid(inputs)

    input_flat=inputs.view(-1)
    target_flat=targets.view(-1)

    intersecion=input_flat*target_flat
    unionsection=input_flat.pow(2).sum()+target_flat.pow(2).sum()+smooth
    loss=unionsection/(2*intersecion.sum()+smooth)
    loss=loss.sum()

    return loss
'''build net work'''
import os
import re
import numpy as np
import torch
from torch import nn
from models.retinanet import resnet34, resnet50

import cfg
from util import load_model


def lr_poly(base_lr, epoch, max_epoch, power):
    return max(0.00000001, base_lr * np.power(1. - epoch / max_epoch, power))


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = None

    if backend.startswith('retinanet'):
        net = resnet50(num_classes=1, pretrained=False)

    if snapshot is not None:
        epoch_counts = re.findall('\d+', snapshot)
        epoch = int(epoch_counts[len(epoch_counts) - 1])
        net = load_model(net, os.path.join(cfg.checkpoint_path, snapshot))
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            net = nn.parallel.DataParallel(net).cuda()
        elif torch.cuda.is_available():
            net = net.cuda()
    return net, epoch


import time
import os
import cv2
import copy
import argparse
import pdb
import collections
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

# import losses
from data_loader import CervicalDataset, collater
from augmentation import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90, Normalizer, UnNormalizer
from torch.utils.data import Dataset, DataLoader

import cfg
from util import *
from losses import FocalLoss
from build_network import build_network, lr_poly
from visdom import Visdom

def train_epoch(model, train_loader, criterion, optimizer, epoch, n_epochs, start_lr, lr_power,
                print_freq=1):
    batch_time = AverageMeter()
    cls_loss_all = AverageMeter()
    reg_loss_all = AverageMeter()
    loss_all = AverageMeter()

    model.train()
    end = time.time()
    np.random.seed()
    for idx, data in enumerate(train_loader):
        annotations = data['label'].cuda()
        classification, regression, anchors = model(data['img'].cuda())
        # print(classification.size())
        cls_loss, reg_loss = criterion(classification, regression, anchors, annotations)
        # print(type(cls_loss),type(reg_loss))
        loss = cls_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_loss_all.update(cls_loss.item(), annotations.size(0))
        reg_loss_all.update(reg_loss.item(), annotations.size(0))
        loss_all.update(loss.item(), annotations.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = lr_poly(start_lr, epoch, n_epochs, lr_power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print stats
        if idx % print_freq == 0:
            res = '\t'.join(['Train, Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                             'Iter: [%d/%d]' % (idx + 1, len(train_loader)),
                             'Time: %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                             'Cls_loss %.4f (%.4f)' % (cls_loss_all.val, cls_loss_all.avg),
                             'Reg_loss %.4f (%.4f)' % (reg_loss_all.val, reg_loss_all.avg),
                             'Loss %.4f (%.4f)' % (loss_all.val, loss_all.avg),
                             'Lr %.8f' % lr,
                             ])
            print(res)

        # update learning rate


    return cls_loss_all.avg, reg_loss_all.avg, loss_all.avg


def train(args):
    model, start_epoch = build_network(snapshot=args.snapshot, backend=cfg.backend)
    optimizer = optim.Adam(model.parameters(), lr=cfg.start_lr, weight_decay=cfg.weight_decay)

    criterion = FocalLoss(alpha=cfg.alpha, gamma=cfg.gamma)


    # viz = Visdom()
    # viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    sample_path = [os.path.join(cfg.train_sample_path, x) for x in os.listdir(cfg.train_sample_path) if '.npz' in x]
    train_data = CervicalDataset(sample_path, cfg.patch_size,
                                 transform=transforms.Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5), RandomRotate90(0.5), Normalizer()]))
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
                              collate_fn=collater, num_workers=cfg.num_worker, worker_init_fn=worker_init_fn)

    epochs = cfg.epochs
    i = 0
    for epoch in range(start_epoch, epochs):
        train_cls_loss, train_reg_loss, train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=epochs,
            start_lr=cfg.start_lr,
            lr_power=cfg.lr_power,
            print_freq=1,
        )
        i += 1
        # viz.line([train_loss], [i], win='train_loss', update='append')
        # time.sleep(0.05)
        save_model(model, epoch, cfg.checkpoint_path)

        mkdir(cfg.log_path)
        with open(cfg.log_path + 'train_log.csv', 'a') as log_file:
            log_file.write(
                '%03d,%0.5f,%0.5f,%0.5f\n' %
                ((epoch + 1),
                 train_cls_loss,
                 train_reg_loss,
                 train_loss)
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default=None,
                        type=str, help='snapshot')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    train(args)

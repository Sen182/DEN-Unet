# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params
import joblib


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='dsb_128_DENUNet_woDS',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args, 3, 1)
    #model = archs.__dict__[args.arch](args)

    model = model.cuda()

    # Data loading code
    img_paths = glob('input/' + args.dataset + '/images/*')
    mask_paths = glob('input/' + args.dataset + '/masks/*')

    train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()

    test_dataset = Dataset(args, test_img_paths, test_mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                target = target.cuda()

                # compute output
                if args.deepsupervision:
                    output = model(input)[-1]
                else:
                    output = model(input)

                output = torch.sigmoid(output).data.cpu().numpy()
                img_paths = test_img_paths[args.batch_size * i:args.batch_size * (i + 1)]

                for i in range(output.shape[0]):
                    imsave('output/%s/' % args.name + os.path.basename(img_paths[i]),
                           (output[i, 0, :, :] * 255).astype('uint8'))

        torch.cuda.empty_cache()

    def dice_coef(pred, target):
        intersection = np.sum(pred * target)
        return 2. * intersection / (np.sum(pred) + np.sum(target) + 1e-5)

    def recall(pred, target):
        intersection = np.sum(pred * target)
        return intersection / (np.sum(target) + 1e-5)

    def accuracy(pred, target):
        tp = np.sum(pred * target)
        tn = np.sum((1 - pred) * (1 - target))
        return (tp + tn) / (np.sum(pred) + np.sum(target) - tp + tn + 1e-5)

    def precision(pred, target):
        tp = np.sum(pred * target)
        fp = np.sum(pred * (1 - target))
        return tp / (tp + fp + 1e-5)
    # IoU
    ious, dice_scores, recall_scores, accuracy_scores, sensitivity_scores, precision_scores = [], [], [], [], [], []
    for i in tqdm(range(len(test_mask_paths))):
        mask = imread(test_mask_paths[i])
        pb = imread('output/%s/'%args.name+os.path.basename(test_mask_paths[i]))

        mask = mask.astype('float32') / 255
        pb = pb.astype('float32') / 255

        '''
        plt.figure()
        plt.subplot(121)
        plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(pb)
        plt.show()
        '''

        iou = iou_score(pb, mask)
        ious.append(iou)

        # 计算 Dice Coefficient
        dice = dice_coef(pb, mask)
        dice_scores.append(dice)

        # 计算 Recall
        rec = recall(pb, mask)
        recall_scores.append(rec)

        # 计算 Accuracy
        acc = accuracy(pb, mask)
        accuracy_scores.append(acc)

        # 计算 Precision
        prec = precision(pb, mask)
        precision_scores.append(prec)
    print('IoU: %.4f' %np.mean(ious))
    print('Dice: %.4f' % np.mean(dice_scores))
    print('Recall: %.4f' % np.mean(recall_scores))
    print('Accuracy: %.4f' % np.mean(accuracy_scores))
    print('Precision: %.4f' % np.mean(precision_scores))


if __name__ == '__main__':
    main()

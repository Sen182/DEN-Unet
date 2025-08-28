import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return self.alpha * bce + (1 - self.alpha) * dice

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha  # 控制 Focal 和 Dice 的权重
        self.gamma = gamma  # 控制 Focal Loss 对难分样本的关注度

    def forward(self, input, target):
        # 计算 Focal BCE Loss
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-bce)  # 计算 pt，衡量样本是否容易分类
        focal_bce = (1 - pt) ** self.gamma * bce  # Focal Loss 公式
        focal_bce = focal_bce.mean()

        # 计算 Dice Loss
        smooth = 1e-5
        input_sigmoid = torch.sigmoid(input)
        num = target.size(0)
        input_sigmoid = input_sigmoid.view(num, -1)
        target = target.view(num, -1)
        intersection = (input_sigmoid * target)
        dice = (2. * intersection.sum(1) + smooth) / (input_sigmoid.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        # 组合损失
        return self.alpha * focal_bce + (1 - self.alpha) * dice

class BCEDiceBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3):
        super(BCEDiceBoundaryLoss, self).__init__()
        self.alpha = alpha  # BCE 和 Dice 的平衡
        self.beta = beta    # 边界损失的权重

    def forward(self, input, target):
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(input, target)

        # Dice Loss
        smooth = 1e-5
        input_sigmoid = torch.sigmoid(input)
        num = target.size(0)
        input_sigmoid = input_sigmoid.view(num, -1)
        target = target.view(num, -1)
        intersection = (input_sigmoid * target)
        dice = (2. * intersection.sum(1) + smooth) / (input_sigmoid.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        # Boundary Loss（利用梯度计算边界）
        dy = torch.abs(target[:, :, 1:] - target[:, :, :-1])  # 垂直边界
        dx = torch.abs(target[:, 1:, :] - target[:, :-1, :])  # 水平边界
        boundary_target = dx + dy  # 计算真实目标的边界

        boundary_pred = torch.abs(input_sigmoid[:, :, 1:] - input_sigmoid[:, :, :-1]) + \
                        torch.abs(input_sigmoid[:, 1:, :] - input_sigmoid[:, :-1, :])
        boundary_loss = F.l1_loss(boundary_pred, boundary_target)  # 计算 L1 边界损失

        return self.alpha * bce + (1 - self.alpha) * dice + self.beta * boundary_loss

class LovaszDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(LovaszDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target):
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(input, target)

        # Lovasz-Softmax Loss
        input_sigmoid = torch.sigmoid(input)
        lovasz = lovasz_hinge(input_sigmoid, target)

        return self.alpha * bce + (1 - self.alpha) * lovasz

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

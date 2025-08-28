# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torch.nn import Transformer

#EfficientNet 基础模块
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, kernel_size, dropout_rate=0.2):
        super(MBConv, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.stride = stride

        self.expand_conv = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.swish = nn.SiLU()

        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.se_conv1 = nn.Conv2d(mid_channels, mid_channels // 8, 1)
        self.se_conv2 = nn.Conv2d(mid_channels // 8, mid_channels, 1)

        self.project_conv = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.spatial_attention = nn.Conv2d(mid_channels, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride,
                                       bias=False) if in_channels != out_channels else nn.Identity()

        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        out = self.swish(self.bn1(self.expand_conv(x)))
        out = self.swish(self.bn2(self.depthwise_conv(out)))

        se_out = F.adaptive_avg_pool2d(out, 1)
        se_out = F.relu(self.se_conv1(se_out))
        se_out = self.sigmoid(self.se_conv2(se_out))
        out = out * se_out

        spatial_attention_map = self.spatial_attention(out)
        out = out * self.sigmoid(spatial_attention_map)

        out = self.bn3(self.project_conv(out))
        out = self.dropout(out)

        if self.use_residual:
            out += self.residual_conv(x)
        return out

class SEACB(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(SEACB, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            #nn.ReLU6(inplace=True)
            nn.SiLU()
        )

        self.block1 = MBConv(32, 16, 1, 1, 3, dropout_rate)
        self.block2 = MBConv(16, 24, 6, 2, 3, dropout_rate)
        self.block3 = MBConv(24, 40, 6, 2, 5, dropout_rate)
        self.block4 = MBConv(40, 80, 6, 2, 3, dropout_rate)
        #self.block5 = MBConv(80, 112, 6, 1, 5)

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        f1 = self.block1(x)   # 16通道
        f2 = self.block2(f1)  # 24通道
        f3 = self.block3(f2)  # 40通道
        f4 = self.block4(f3)  # 80通道
        #f5 = self.block5(f4)  # 112通道

        f4 = self.dropout(f4)

        return [f1, f2, f3, f4]  # 返回特征图


class DynamicConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DynamicConvolution, self).__init__()
        # 通过MLP学习生成动态卷积核
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * kernel_size * kernel_size, kernel_size=1, bias=False),
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # 学习到的卷积核
        batch_size, channels, height, width = x.size()
        dynamic_kernels = self.fc(x)
        dynamic_kernels = dynamic_kernels.view(batch_size, channels, self.kernel_size, self.kernel_size, height, width)

        # 使用动态卷积
        out = F.conv2d(x, dynamic_kernels, stride=self.stride, padding=self.padding, groups=channels)
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, height * width)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        # 计算自注意力权重
        attention_map = torch.bmm(query.permute(0, 2, 1), key)
        attention_map = F.softmax(attention_map, dim=-1)

        # 计算加权的value
        out = torch.bmm(value, attention_map.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        return self.gamma * out + x

class NonLocalAttention(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Flatten the spatial dimensions
        query = self.query_conv(x).view(batch_size, -1, height * width)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        # Compute attention
        attention = torch.bmm(query.transpose(1, 2), key)  # (B, H*W, H*W)
        attention = self.softmax(attention)

        # Apply attention to value
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, channels, height, width)

        return out

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels*2, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels*2, 1, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels*2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.conv_out = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.max_pool(x)
        x2 = self.avg_pool(x)
        x = torch.cat([x1, x2], dim=1)
        attn1 = self.conv1(x)
        attn2 = self.conv2(x)
        attn3 = self.conv3(x)
        attention_map = self.bn(torch.cat([attn1, attn2, attn3], dim=1))
        attention_map = self.conv_out(attention_map)
        return self.sigmoid(attention_map)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x))))
        attention = self.bn(avg_out + max_out)
        return self.sigmoid(attention)

class FusionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(FusionAttention, self).__init__()
        self.spatial_attention = SpatialAttention(in_channels)
        self.channel_attention = ChannelAttention(in_channels, reduction)

    def forward(self, x):
        # 计算空间注意力和通道注意力
        spatial_att = self.spatial_attention(x)
        channel_att = self.channel_attention(x)

        # 将空间和通道注意力进行融合
        return x * (spatial_att + channel_att)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=6, padding=6)
        self.conv12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=12, padding=12)
        self.conv18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=18, padding=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv_global = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        self.concat_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        # 全局平均池化后的处理
        global_avg = self.global_avg_pool(x)
        global_avg = self.conv_global(global_avg)
        global_avg = F.interpolate(global_avg, size=x.size()[2:], mode='bilinear', align_corners=True)

        # 其他不同膨胀率的卷积
        x1 = self.conv1(x)
        x6 = self.conv6(x)
        x12 = self.conv12(x)
        x18 = self.conv18(x)

        # 合并所有特征图
        x_out = torch.cat([x1, x6, x12, x18, global_avg], dim=1)
        x_out = self.concat_conv(x_out)

        return x_out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(), stride=1, dilation=2):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            '''self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )'''

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.act_func(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.act_func(out2)

        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = self.act_func(out3)
        #resnet结构
        out3 += self.shortcut(x)
        out3 = self.act_func(out3)

        return out3

class ASPPVGGBlock1(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(), stride=1, dilation=2):
        super(ASPPVGGBlock1, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.aspp4 = ASPP(nb_filter[1], nb_filter[1])


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            '''self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )'''

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.act_func(out1)

        out1 = self.aspp4(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.act_func(out2)


        #resnet结构
        out2 += self.shortcut(x)
        out2 = self.act_func(out2)

        return out2

class ASPPVGGBlock2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(), stride=1, dilation=2):
        super(ASPPVGGBlock2, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.aspp4 = ASPP(nb_filter[2], nb_filter[2])


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            '''self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )'''

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.act_func(out1)

        out1 = self.aspp4(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.act_func(out2)


        #resnet结构
        out2 += self.shortcut(x)
        out2 = self.act_func(out2)

        return out2

class ASPPVGGBlock3(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(), stride=1, dilation=2):
        super(ASPPVGGBlock3, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.aspp4 = ASPP(nb_filter[3], nb_filter[3])


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            '''self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )'''

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.act_func(out1)

        out1 = self.aspp4(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.act_func(out2)


        #resnet结构
        out2 += self.shortcut(x)
        out2 = self.act_func(out2)

        return out2

class ASPPVGGBlock4(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(), stride=1, dilation=2):
        super(ASPPVGGBlock4, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.aspp4 = ASPP(nb_filter[4], nb_filter[4])


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            '''self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )'''

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.act_func(out1)

        out1 = self.aspp4(out1)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.act_func(out2)


        #resnet结构
        out2 += self.shortcut(x)
        out2 = self.act_func(out2)

        return out2

class DENUNet(nn.Module):
    def __init__(self, args, in_chns, class_num):
        super().__init__()

        self.args = args

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #添加 EfficientNet 作为第二编码器
        self.SEACB = SEACB()
        #self.eff_conv0 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.eff_conv1 = nn.Conv2d(48, nb_filter[0], kernel_size=1, stride=1, padding=0)
        self.eff_conv2 = nn.Conv2d(88, nb_filter[1], kernel_size=1, stride=1, padding=0)
        self.eff_conv3 = nn.Conv2d(168, nb_filter[2], kernel_size=1, stride=1, padding=0)
        self.eff_conv4 = nn.Conv2d(336, nb_filter[3], kernel_size=1, stride=1, padding=0)

        # **1x1 卷积调整 EfficientNet 输出通道数**
        '''self.eff_conv_adapt = nn.ModuleList([
            nn.Conv2d(16, nb_filter[0], kernel_size=1),
            nn.Conv2d(24, nb_filter[1], kernel_size=1),
            nn.Conv2d(40, nb_filter[2], kernel_size=1),
            nn.Conv2d(80, nb_filter[3], kernel_size=1),
            nn.Conv2d(112, nb_filter[4], kernel_size=1),
        ])'''

        # 添加FusionAttention
        '''self.fa0_1 = FusionAttention(in_channels=nb_filter[0])
        self.fa0_2 = FusionAttention(in_channels=nb_filter[0]*2)
        self.fa0_3 = FusionAttention(in_channels=nb_filter[0]*3)
        self.fa0_4 = FusionAttention(in_channels=nb_filter[0]*4)
        self.fa1_1 = FusionAttention(in_channels=nb_filter[1])
        self.fa1_2 = FusionAttention(in_channels=nb_filter[1]*2)
        self.fa1_3 = FusionAttention(in_channels=nb_filter[1]*3)
        self.fa2_1 = FusionAttention(in_channels=nb_filter[2])
        self.fa2_2 = FusionAttention(in_channels=nb_filter[2]*2)
        self.fa3_1 = FusionAttention(in_channels=nb_filter[3])'''

        self.fa0 = FusionAttention(in_channels=nb_filter[0])
        self.fa1 = FusionAttention(in_channels=nb_filter[1])
        self.fa2 = FusionAttention(in_channels=nb_filter[2])
        self.fa3 = FusionAttention(in_channels=nb_filter[3])

        self.conv0_0 = VGGBlock(args.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ASPPVGGBlock1(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ASPPVGGBlock2(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ASPPVGGBlock3(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ASPPVGGBlock4(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.args.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, input):

        eff_features = self.SEACB(input)
        #eff_features = [self.eff_conv_adapt[i](feat) for i, feat in enumerate(eff_features)]  # 1x1 conv 适配通道
        '''for i, feat in enumerate(eff_features):
            print(f"EfficientNet Feature {i}: {feat.shape}")'''

        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.eff_conv1(torch.cat([self.pool(x0_0), eff_features[0]], dim=1)))
        x0_1 = self.conv0_1(torch.cat([self.fa0(x0_0), self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.eff_conv2(torch.cat([self.pool(x1_0), eff_features[1]], dim=1)))
        x1_1 = self.conv1_1(torch.cat([self.fa1(x1_0), self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([self.fa0(x0_0), self.fa0(x0_1), self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.eff_conv3(torch.cat([self.pool(x2_0), eff_features[2]], dim=1)))
        x2_1 = self.conv2_1(torch.cat([self.fa2(x2_0), self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([self.fa1(x1_0), self.fa1(x1_1), self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([self.fa0(x0_0), self.fa0(x0_1), self.fa0(x0_2), self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.eff_conv4(torch.cat([self.pool(x3_0), eff_features[3]], dim=1)))
        x3_1 = self.conv3_1(torch.cat([self.fa3(x3_0), self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([self.fa2(x2_0), self.fa2(x2_1), self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([self.fa1(x1_0), self.fa1(x1_1), self.fa1(x1_2), self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([self.fa0(x0_0), self.fa0(x0_1), self.fa0(x0_2), self.fa0(x0_3), self.up(x1_3)], 1))

        if self.args.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

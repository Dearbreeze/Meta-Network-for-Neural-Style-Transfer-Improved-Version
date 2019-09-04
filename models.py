import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from collections import defaultdict

from utils import *




class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


class MyConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation = 1):
        super(MyConv2D, self).__init__()
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size)).cuda()
        self.bias = torch.zeros(out_channels).cuda()
        self.dilation = dilation
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride ,padding = self.padding,dilation = self.dilation)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        return s.format(**self.__dict__)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1),
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.conv(x) + x

class ConvLayer1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1):
        super(ConvLayer1, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation= dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class BasicBlock_Dilated2(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Dilated2, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(in_planes, planes, kernel_size=1, stride=stride),
            *ConvLayer(planes, planes, kernel_size=2, stride=stride, dilation=2),
            *ConvLayer(planes, planes, kernel_size=1, stride=stride),
        )


    def forward(self, x):
        out = self.conv(x)
        return out + x

def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,dilation = 1,ReflectionPad = True,padding = 0,
              upsample=None, instance_norm=True, relu=True, trainable=False):
    layers = []
    if upsample:
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))
    if ReflectionPad:
        layers.append(nn.ReflectionPad2d(kernel_size // 2))
    if trainable:
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    else:
        layers.append(MyConv2D(in_channels, out_channels, kernel_size, stride,padding,dilation))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())
    return layers


class TransformNet(nn.Module):
    def __init__(self, base=8):
        super(TransformNet, self).__init__()
        self.base = base
        self.weights = []
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9, trainable=True),
            *ConvLayer(base, base * 2, kernel_size=3, stride=2),
            *ConvLayer(base * 2, base * 4, kernel_size=3, stride=2),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(base * 4) for i in range(5)])
        self.upsampling = nn.Sequential(
            *ConvLayer(base * 4, base * 2, kernel_size=3, upsample=2),
            *ConvLayer(base * 2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False, trainable=True),
        )
        self.get_param_dict()

    def forward(self, X):
        #print(X.size())
        y = self.downsampling(X)
        #print(y.size())
        y = self.residuals(y)
        y = self.upsampling(y)
        #print(y.size())
        return y

    def get_param_dict(self):
        """找出该网络所有 MyConv2D 层，计算它们需要的权值数量"""
        param_dict = defaultdict(int)

        def dfs(module, name):
            for name2, layer in module.named_children():
                dfs(layer, '%s.%s' % (name, name2) if name != '' else name2)
            if module.__class__ == MyConv2D:
                param_dict[name] += int(np.prod(module.weight.shape))
                param_dict[name] += int(np.prod(module.bias.shape))

        dfs(self, '')
        return param_dict

    def set_my_attr(self, name, value):
        # 下面这个循环是一步步遍历类似 residuals.0.conv.1 的字符串，找到相应的权值
        target = self
        for x in name.split('.'):
            if x.isnumeric():
                target = target.__getitem__(int(x))
            else:
                target = getattr(target, x)

        # 设置对应的权值
        n_weight = np.prod(target.weight.shape)
        target.weight = value[:n_weight].view(target.weight.shape)
        target.bias = value[n_weight:].view(target.bias.shape)

    def set_weights(self, weights, i=0):
        """输入权值字典，对该网络所有的 MyConv2D 层设置权值"""
        for name, param in weights.items():
            self.set_my_attr(name, weights[name][i])


class MetaNet(nn.Module):
    def __init__(self, param_dict):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(76128, 128 * self.param_num)
        self.pooling = nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))

    def forward(self, gram_features):
        gram_value_feature = []
        gram_features1 = gram_features.copy()
        gram_features1[3] = self.pooling(gram_features1[3])
        '''
        for x in gram_features1:
            x = x.view(1,-1)
            gram_value_feature.append(x)
        gram_value_feature = torch.cat(gram_value_feature,dim=-1)
        '''
        for x in gram_features1:
            size = x.size()[1]
            for y in range(size):
                cut_matrix = x[0][y][y:]
                gram_value_feature.extend(cut_matrix)
        gram_value_feature = torch.Tensor(gram_value_feature).unsqueeze(0).cuda()

        hidden = F.relu(self.hidden(gram_value_feature))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])


        return filters
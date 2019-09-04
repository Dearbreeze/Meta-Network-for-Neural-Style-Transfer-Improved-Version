import os


import torchnet as tnt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
from PIL import Image
import matplotlib.pyplot as plt
import torchvision as tv
import torchvision
import torchvision.transforms as transforms
import shutil
from glob import glob

from tensorboardX import SummaryWriter

import numpy as np
import multiprocessing
import time
import copy
from tqdm import tqdm
from collections import defaultdict


import torch.utils.data.distributed

from example.utils import *
from example.models import *
import time
import warnings
from pprint import pprint

warnings.filterwarnings("ignore")
# In[5]:

class opting(object):
    image_size = 256  # 图片大小
    batch_size = 8
    content_data_root = '/home/lyx/PycharmProjects/Dear/styletransformdata/content/'  # 数据集存放路径
    style_data_root = '/home/lyx/PycharmProjects/Dear/styletransformdata/style/'
    num_workers = 4  # 多线程加载数据
    use_gpu = True  # 使用GPU
    width = 256
    base = 32
    lr = 1e-3
    env = 'metanetwork-style-1-2e5-test28'  # visdom env
    plot_every = 10  # 每10个batch可视化一次
    epochs = 18    # 训练epoch
    epoch_vgg = 2
    tv_weight = 1e-6
    content_weight = 1  # content_loss 的权重    最后content_loss = 22
    #style_weight = 60  # style_loss的权重          最后style_loss = 8
    gram_weight = 2e5
    metanet_model_path = './checkpoints/metanet-1-2e5-test28.pth'  # 预训练模型的路径
    transform_net_model_path = './checkpoints/transform_net-1-2e5-test28.pth'  # 预训练模型的路径


    style_path = './style.jpg'  # 风格图片存放路径
    content_path = './content.jpg'  # 需要进行分割迁移的图片
    result_path = 'content-1-2e5-test28.jpg'  # 风格迁移结果的保存路径


opt = opting()
# # 搭建模型



vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).cuda().eval()



transform_net = TransformNet(opt.base).cuda()
transform_net.get_param_dict()
#print(transform_net.get_param_dict())

metanet = MetaNet(transform_net.get_param_dict()).cuda()


opt = opting()
vis = Visualizer(opt.env)          #可视化

# # 载入数据集
#
# > During training, each content image or style image is resized to keep the smallest dimension in the range [256, 480], and randomly cropped regions of size 256 × 256.
#
# ## 载入 COCO 数据集和 WikiArt 数据集
#
# > The batch size of content images is 8 and the meta network is trained for 20 iterations before changing the style image.




data_transform = transforms.Compose([
    transforms.RandomResizedCrop(opt.width, scale=(256 / 480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
    tensor_normalizer
])

style_dataset = torchvision.datasets.ImageFolder(opt.style_data_root, transform=data_transform)
content_dataset = torchvision.datasets.ImageFolder(opt.content_data_root, transform=data_transform)

content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=opt.batch_size,
                                                  shuffle=True, num_workers=opt.num_workers)



trainable_params = {}
trainable_param_shapes = {}
for model in [ vgg16,transform_net, metanet]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params[name] = param
            trainable_param_shapes[name] = param.shape
# # 开始训练

# In[ ]:


optimizer = optim.Adam(trainable_params.values(), 1e-3)



n_batch = len(content_dataset)
metanet.train()
transform_net.train()

# 损失统计
#style_meter = tnt.meter.AverageValueMeter()
content_meter = tnt.meter.AverageValueMeter()
gram_meter = tnt.meter.AverageValueMeter()

'''
def per_train():   #content_loss
    for epoch in range(opt.epoch_vgg):
        content_meter.reset()
        with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
            for batch, (content_images, _) in pbar:
                optimizer.zero_grad()
                if batch % 40 == 0:
                    style_image = random.choice(style_dataset)[0].unsqueeze(0).cuda()
                    style_features = vgg16(style_image)

                # 使用风格图像生成风格模型
                weights = metanet(mean_std(style_features))
                transform_net.set_weights(weights, 0)  ##只计算均值和标准差，不计算 Gram 矩阵,来生成权重

                # 使用风格模型预测风格迁移图像
                content_images = content_images.cuda()
                transformed_images = transform_net(content_images)

                # 使用 vgg16 计算特征
                content_features = vgg16(content_images)
                transformed_features = vgg16(transformed_images)

                content_loss = opt.content_weight * F.mse_loss(transformed_features[2], content_features[2])
                content_loss.backward()
                optimizer.step()
                content_meter.add(float(content_loss.data))
                if batch % opt.plot_every == 0:
                    vis.plot('content_loss', content_meter.value()[0])

'''

def train():
    for epoch in range(opt.epochs):
        content_meter.reset()
        #style_meter.reset()
        gram_meter.reset()

        with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
            for batch, (content_images, _) in pbar:


                # 每 20 个 batch 随机挑选一张新的风格图像，计算其特征
                if batch % 20 == 0:
                    style_image = random.choice(style_dataset)[0].unsqueeze(0).cuda()
                    #style_features = vgg16(style_image)
                    #style_mean_std = mean_std(style_features)

                    # 风格图片的gram矩阵
                    style_v = Variable(style_image, volatile=True)
                    features_style = vgg16(style_v)
                    #print('a',features_style)
                    gram_style = [Variable(gram_matrix(y.data)) for y in features_style]

                    half_gram = [half_gram_matrix(k) for k in gram_style]
                    #print(half_gram)

                    vis.img('style', (style_image.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                # 检查纯色
                x = content_images.cpu().numpy()
                if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                    continue

                optimizer.zero_grad()

                # 使用风格图像生成风格模型

                weights = metanet(half_gram)
                transform_net.set_weights(weights, 0)        ##只计算均值和标准差，不计算 Gram 矩阵,来生成权重

                # 使用风格模型预测风格迁移图像
                content_images = content_images.cuda()
                transformed_images = transform_net(content_images)
                #print('b',transformed_images)

                # 使用 vgg16 计算特征
                content_features = vgg16(content_images)
                transformed_features = vgg16(transformed_images)
                #print('c',transformed_features)

                # content loss
                content_loss = opt.content_weight * F.mse_loss(transformed_features[2], content_features[2])
                '''
                # style loss
                style_loss = opt.style_weight * F.mse_loss(transformed_mean_std,
                                                       style_mean_std.expand_as(transformed_mean_std))
                
                # total variation loss
                y = transformed_images
                tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                       torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))
                '''

                # gram loss
                gram_loss = 0
                for ft_y, gm_s in zip(transformed_features, gram_style):
                    gram_y = gram_matrix(ft_y)
                    gram_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
                gram_loss *= opt.gram_weight


                # 求和
                loss = content_loss + gram_loss

                content_meter.add(float(content_loss.data))
                #style_meter.add(float(style_loss.data))
                gram_meter.add(float(gram_loss.data))

                loss.backward()
                optimizer.step()


                if batch % opt.plot_every == 0:
                    print('epoch：{},loss:{}, content_loss{}, gram_loss{}'.format(epoch, loss, content_loss, gram_loss))

                    # 可视化
                    vis.plot('gram_loss', gram_meter.value()[0])
                    vis.plot('content_loss', content_meter.value()[0])
                    #vis.plot('style_loss', style_meter.value()[0])
                    vis.img('output', (transformed_images.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                    vis.img('input', (content_images.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

                del transformed_images, weights
        # 保存visdom和模型
        vis.save([opt.env])
        torch.save(metanet.state_dict(),opt.metanet_model_path )
        torch.save(transform_net.state_dict(), opt.transform_net_model_path)

def stylise():
    metanet.load_state_dict(torch.load(opt.metanet_model_path))
    transform_net.load_state_dict(torch.load(opt.transform_net_model_path))

    style_image = tv.datasets.folder.default_loader(opt.style_path)
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    normalize = transforms.Compose([
        transforms.RandomResizedCrop(opt.width, scale=(256 / 480, 1), ratio=(1, 1)),
        transforms.ToTensor(),
        tensor_normalizer,
            ])
    style_image = normalize(style_image).unsqueeze(0).cuda()
    content_image = normalize(content_image).unsqueeze(0).cuda()
    #torchvision.utils.save_image((content_image.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1), './a.jpg')
    style_v = Variable(style_image, volatile=True)
    features_style = vgg16(style_v)
    gram_style = [Variable(gram_matrix(y.data)) for y in features_style]


    weights = metanet(gram_style)
    transform_net.set_weights(weights, 0)  ##只计算均值和标准差，不计算 Gram 矩阵,来生成权重

    transformed_image = transform_net(content_image)

    torchvision.utils.save_image((transformed_image.data.cpu()[0]* 0.225 + 0.45).clamp(min=0,max=1),opt.result_path)

if __name__ == '__main__':

    train()
    stylise()
    '''
    t1 = time.time()
    X = torch.rand((1,3,256,256)).cuda()
    for i in range(1000):
        feature = vgg16(X)
        gram_style = [Variable(gram_matrix(y.data)) for y in feature]
        weights = metanet(gram_style)
        transform_net.set_weights(weights, 0)
        transform_net(X)
        del feature,weights,gram_style

    t2 = time.time()
    print(t2-t1)
    '''


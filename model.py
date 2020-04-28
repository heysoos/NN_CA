import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from PIL import Image
import cv2
from skimage.color import rgba2rgb
from skimage.io import imread

from os import makedirs, path
from shutil import copyfile

from collections import OrderedDict


# CHANNEL_N = 16
# RADIUS = 1
# NUM_FILTERS = 10
# HIDDEN_N = 128
#
# EMBED_KERNEL = 5

TARGET_SIZE = 40
TARGET_PADDING = 16
BATCH_SIZE = 1


class Filter(nn.Module):
    def __init__(self, r, symmetric=True):
        super().__init__()

        f = torch.randn(2 * r + 1)
        if symmetric:
            f = (f + f.flip(0)) / 2
        f = torch.ger(f, f)
        #         f[r, r] = 0
        f = f - f.mean()
        #         f = f - f.sum()/(f.numel() - 1)

        f[r, r] = 0
        f = f / (f.numel() - 1)

        #         f = f / (f.numel())

        self.kernel = nn.Parameter(f)


class CAModel(nn.Module):
    def __init__(self, channel_n, r, num_filters, hidden_n):
        super().__init__()

        self.channel_n = channel_n
        self.r = r
        self.num_filters = num_filters
        self.fire_rate = 0.5

        # define identity matrix
        identity = torch.zeros(2 * self.r + 1)
        identity[self.r] = 1
        self.identity = torch.ger(identity, identity)
        self.identity = nn.Parameter(self.identity, requires_grad=False)

        # initialize perception kernel (trainable)
        self.rand_filters = [Filter(self.r, symmetric=True).kernel for i in range(self.num_filters)]

        self.filters = nn.ParameterList([self.identity] + self.rand_filters)

        # Sobel filters
        #         self.sx = torch.ger(torch.FloatTensor([1, 2, 1]), torch.FloatTensor([-1, 0, 1])) / 8
        #         self.sx = nn.Parameter(self.sx, requires_grad=False)
        #         self.sy = nn.Parameter(self.sx.T.type(torch.FloatTensor), requires_grad=False)
        #         self.filters = [torch.cuda.FloatTensor(self.sx), torch.cuda.FloatTensor(self.sy)]

        # 1D conv network
        self.dmodel = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.channel_n * (self.num_filters + 1), hidden_n, 1, padding_mode='circular')),
            ('relu1', nn.LeakyReLU()),
            ('conv2', nn.Conv2d(hidden_n, self.channel_n, 1, padding_mode='circular')),
        ]))

        # 1D conv network
        #         self.dmodel = [nn.Conv2d(self.channel_n*(self.num_filters + 1), hidden_n[0], 1, padding_mode='circular')]
        #         self.dmodel.append(nn.LeakyReLU())
        #         for i in range(1, len(hidden_n)):
        #             l = nn.Conv2d(hidden_n[i-1], hidden_n[i], 1, padding_mode='circular')
        #             self.dmodel.append(l)
        #             if i < (len(hidden_n) - 1):
        #                 self.dmodel.append(nn.LeakyReLU())

        gain = nn.init.calculate_gain('leaky_relu')
        #         for l in self.dmodel:
        #             if type(l) == nn.Conv2d:
        #                 nn.init.xavier_uniform_(l.weight, gain)
        #         self.dmodel = nn.Sequential(*self.dmodel)

        # update rule - initialized with zeros so initial behaviour is 'do nothing' (trainable)
        #         gain=nn.init.calculate_gain('relu')
        #         nn.init.zeros_(self.dmodel.conv2.weight)
        #         nn.init.zeros_(self.dmodel.conv2.bias)

        nn.init.xavier_uniform_(self.dmodel.conv1.weight, gain)
        #         nn.init.zeros_(self.dmodel.conv2.bias)

        nn.init.xavier_uniform_(self.dmodel.conv2.weight, gain)

    def perceive(self, x):
        #         filters = [self.identity] + [f.kernel for f in self.rand_filters]
        #         filters = [self.identity, self.sx, self.sy]
        filters = [f for f in self.filters]
        numFilters = len(filters)
        k_size = 2 * self.r + 1

        filters = torch.stack(filters).unsqueeze(0)
        filters = torch.repeat_interleave(filters, self.channel_n, dim=0)
        filters = filters.view(self.channel_n * numFilters, 1, k_size,
                               k_size)  # combine filters into batch dimension (or out dimension, idk)

        # depthwise conv2d (groups==self.channel_n)
        x = F.pad(x, (self.r, self.r, self.r, self.r), mode='circular')
        y = F.conv2d(x, filters, padding=self.r, groups=self.channel_n)
        y = y[:, :, self.r:-self.r, self.r:-self.r]
        return y

    def get_living_mask(self, x):
        alpha_channel = x[:, 3:4, :, :]
        alpha_channel = F.pad(alpha_channel, (self.r, self.r, self.r, self.r), mode='circular')

        alive_mask = F.max_pool2d(alpha_channel, kernel_size=2 * self.r + 1, stride=1, padding=self.r) > 0.1
        alive_mask = alive_mask[:, :, self.r:-self.r, self.r:-self.r]

        death_mask = F.avg_pool2d(alpha_channel, kernel_size=2 * self.r + 1, stride=1, padding=self.r) < 0.2
        death_mask = death_mask[:, :, self.r:-self.r, self.r:-self.r]
        return alive_mask.cuda() & death_mask.cuda()

    def forward(self, x, fire_rate=None, step_size=1.0):
        pre_life_mask = self.get_living_mask(x)

        y = self.perceive(x)
        dx = self.dmodel(y) * step_size

        if fire_rate is None:
            fire_rate = self.fire_rate
        update_mask = (torch.rand(x[:, :1, :, :].shape) <= fire_rate).type(torch.FloatTensor).cuda()
        x = x + dx * update_mask

        post_life_mask = self.get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).type(torch.FloatTensor).cuda()

        x = x * life_mask

        return x

class Embedder(nn.Module):
    def __init__(self, embed_kernel):
        super().__init__()

        self.c1 = nn.Conv2d(4, 32, embed_kernel, padding=embed_kernel - 1, padding_mode='circular',
                            stride=embed_kernel // 2)
        self.c2 = nn.Conv2d(32, 32, embed_kernel, padding=embed_kernel - 1, padding_mode='circular',
                            stride=embed_kernel // 2)
        mp_kernel = 5  # SIZE / (mp_kernel*2) / mp_kernal = 1
        self.mp1 = nn.MaxPool2d(3, stride=2)

        self.c3 = nn.Conv2d(32, 32, embed_kernel, padding=embed_kernel - 1, padding_mode='circular',
                            stride=embed_kernel // 2)
        self.c4 = nn.Conv2d(32, 8, embed_kernel, padding=embed_kernel - 1, padding_mode='circular')
        self.mp2 = nn.AvgPool2d(3, stride=2)

        # average pooling

    def forward(self, x):
        z = F.leaky_relu(self.c1(x))
        z = F.leaky_relu(self.c2(z))
        z = self.mp1(z)

        z = F.leaky_relu(self.c3(z))
        z = self.c4(z)
        z = self.mp2(z)

        return z


def to_rgba(x):
    return x[:, :4, :, :]


def to_alpha(x):
    return np.clip(x[:, 3:4, :, :], 0.0, 1.0)


def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[:, :3, :, :], to_alpha(x)
    return 1.0 - a + rgb


def load_image(fname, max_size=TARGET_SIZE):
    img = Image.open(fname)
    img.thumbnail((max_size, max_size), Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def copy_model(folder):
    src = 'model.py'
    dst = path.join(folder, src)
    copyfile(src, dst)


# def save_settings(folder, settings):
#     with open(folder + 'settings.csv', 'w') as f:
#         for key in settings.keys():
#             f.write("%s,%s\n" % (key, settings[key]))
#     pickle_out = open('{}settings.pickle'.format(folder), 'wb')
#     pickle.dump(settings, pickle_out)
#     pickle_out.close()
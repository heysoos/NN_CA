import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from math import *
import time

HIDDEN = 8
DIM = 8
RES = 64


def sharpen(x, alpha):
    # Expects dim x XR x YR
    lx = torch.log(1e-8 + x)
    # H = -torch.sum(x*lx,0).unsqueeze(0)
    # lx = lx*(1+H)
    return F.softmax(alpha * lx, dim=0)


def totalistic(x):
    z = 0.125 * (x + x.flip(2) + x.flip(3) + x.flip(2).flip(3))
    z = z + 0.125 * (x.transpose(2, 3) + x.transpose(2, 3).flip(2) + x.transpose(2, 3).flip(3) + x.transpose(2, 3).flip(
        2).flip(3))
    z = z - z.mean(3).mean(2).unsqueeze(2).unsqueeze(3)

    return z


class Rule(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter1 = nn.Parameter(torch.randn(HIDDEN, 1, 5, 5))
        self.bias1 = nn.Parameter(0 * torch.randn(HIDDEN))

        self.filter2 = nn.Conv2d(HIDDEN, HIDDEN, 1, padding=0)
        self.filter3 = nn.Conv2d(HIDDEN, HIDDEN, 1, padding=0)


class CA(nn.Module):
    def __init__(self):
        super().__init__()

        self.rule = Rule()
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-5)

    def initGrid(self, BS):
        self.psi = torch.cuda.FloatTensor(2 * np.random.rand(BS, HIDDEN, RES, RES) - 1)

    def forward(self):
        z = F.conv2d(self.psi, weight=totalistic(self.rule.filter1), bias=self.rule.bias1, padding=2, groups=HIDDEN)
        z = F.leaky_relu(z)
        z = F.leaky_relu(self.rule.filter2(z))
        self.psi = torch.tanh(self.psi + self.rule.filter3(z))

    def cleanup(self):
        del self.psi


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(1, 32, 3, padding=1)
        nn.init.orthogonal_(self.c1.weight, gain=sqrt(2))
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        nn.init.orthogonal_(self.c2.weight, gain=sqrt(2))
        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        nn.init.orthogonal_(self.c3.weight, gain=sqrt(2))
        self.c4 = nn.Conv2d(32, 32, 3, padding=1)
        nn.init.orthogonal_(self.c4.weight, gain=sqrt(2))
        self.c5 = nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, x):
        z = F.leaky_relu(self.c1(x[:, 0:1, :, :]))
        z = F.leaky_relu(self.c2(z))
        z = F.leaky_relu(self.c3(z))
        z = F.leaky_relu(self.c4(z))
        z = self.c5(z)

        return z


def findHardNegative(zs):
    step = 0

    while step < 1000:
        i = np.random.randint(zs.shape[0])
        j = i
        k = np.random.randint(zs.shape[0] - 1)
        if k >= i:
            k += 1

        i2 = np.random.randint(zs.shape[1])
        j2 = np.random.randint(zs.shape[1] - 1)
        if j2 >= i2:
            j2 += 1
        k2 = np.random.randint(zs.shape[1])

        z1 = zs[i, i2]
        z2 = zs[j, j2]
        z3 = zs[k, k2]

        delta = np.sqrt(np.sum((z1 - z2) ** 2, axis=0)) - np.sqrt(np.sum((z1 - z3) ** 2, axis=0)) + 1
        if delta >= 0.9:
            break
        step += 1

    return i, k, step


population = [CA().cuda() for i in range(1000)]

"""
for i in range(1,len(population)):
    population[i].rule.filter1.data = population[0].rule.filter1.data.detach()
    population[i].rule.bias1.data = population[0].rule.bias1.data.detach()
    population[i].rule.filter2.weight.data = population[0].rule.filter2.weight.data.detach()
    population[i].rule.filter3.weight.data = population[0].rule.filter3.weight.data.detach()
    population[i].rule.filter2.bias.data = population[0].rule.filter2.bias.data.detach()
    population[i].rule.filter3.bias.data = population[0].rule.filter3.bias.data.detach()
"""

embed = Embedder().cuda()

emb_err = []
ca_err = []
hard_frac = []

e_optim = torch.optim.Adam(embed.parameters(), lr=1e-4)
tloss = nn.TripletMarginLoss()

CBS = 5
EBS = 60
ESTEPS = 100

for epoch in range(10000):
    e_loss = []
    c_loss = []
    h_loss = []

    zs = []
    xs = []
    with torch.no_grad():
        for ca in population:
            lzs = []
            ca.initGrid(CBS)
            for j in range(25):
                ca.forward()
            lzs.append(embed.forward(ca.psi).mean(3).mean(2).cpu().detach().numpy())
            # lxs.append(ca.psi.cpu().detach().numpy())
            zs.append(np.concatenate(lzs, axis=0)[np.newaxis])
            ca.cleanup()
        zs = np.concatenate(zs, axis=0)

    # Train embedder
    for i in range(ESTEPS):
        x1 = []
        x2 = []
        x3 = []

        for j in range(EBS):
            a, b, hard = findHardNegative(zs)

            CA1 = population[a]
            CA2 = population[b]

            with torch.no_grad():
                CA1.initGrid(2)
                CA2.initGrid(1)

                K1 = 25
                K2 = 25

                for k in range(K1):
                    CA1.forward()
                for k in range(K2):
                    CA2.forward()

                im1 = CA1.psi[0].cpu().detach().numpy()
                im2 = CA1.psi[1].cpu().detach().numpy()
                im3 = CA2.psi[0].cpu().detach().numpy()

                CA1.cleanup()
                CA2.cleanup()

            x1.append(im1)
            x2.append(im2)
            x3.append(im3)
            h_loss.append(hard)

        x1 = torch.cuda.FloatTensor(np.array(x1))
        x2 = torch.cuda.FloatTensor(np.array(x2))
        x3 = torch.cuda.FloatTensor(np.array(x3))

        e_optim.zero_grad()
        z1 = embed.forward(x1)
        z2 = embed.forward(x2)
        z3 = embed.forward(x3)

        loss = tloss(z1, z2, z3)
        loss.backward()
        e_optim.step()

        e_loss.append(loss.cpu().detach().item())

    emb_err.append(np.mean(e_loss))
    hard_frac.append(np.mean(h_loss))

    # Train CAs
    for ii in range(len(population) // 2):
        i = np.random.randint(len(population))
        CA1 = population[i]
        z1 = zs[i, 0]
        z2 = zs[i, 1]

        d12 = np.sqrt(np.sum((z1 - z2) ** 2, axis=0))
        step = 0
        while step < 1000:
            j = np.random.randint(len(population) - 1)
            if j >= i:
                j += 1

            z3 = zs[j, 0]

            d13 = np.sqrt(np.sum((z1 - z3) ** 2, axis=0))
            if d12 - d13 + 1 > 1:
                break

            step += 1

        CA2 = population[j]

        CA1.optim.zero_grad()
        CA2.optim.zero_grad()

        CA1.initGrid(CBS)
        CA2.initGrid(CBS)

        S1 = 25
        S2 = 25

        for j in range(S1):
            CA1.forward()

        for j in range(S2):
            CA2.forward()

        im = CA1.psi.detach().cpu().numpy()[0, 0, :, :] * 0.5 + 0.5
        im = (255 * np.clip(im, 0, 1)).astype(np.uint8)
        im = Image.fromarray(im)
        im.save("/sata/frames/%.6d.png" % i)

        z1 = embed.forward(CA1.psi)
        z2 = embed.forward(CA2.psi)

        loss = -torch.sqrt(1e-8 + torch.sum((z1 - z2) ** 2, 1)).mean()
        loss.backward()
        CA1.optim.step()
        CA2.optim.step()

        c_loss.append(loss.cpu().detach().item())

        CA1.cleanup()
        CA2.cleanup()

    ca_err.append(np.mean(c_loss))

    np.savetxt("embed.txt", np.array(emb_err))
    np.savetxt("ca.txt", np.array(ca_err))
    np.savetxt("hard.txt", np.array(hard_frac))


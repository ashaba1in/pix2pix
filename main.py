import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms

import os
from PIL import Image
import numpy as np

import time

from model import Pix2pix
from dataset import Dataset


def train(loader, model, loss='gan'):
    global step

    model.train()

    g_epoch_loss, d_epoch_loss = 0.0, 0.0
    for x, y in loader:
        model.step += 1
        g_loss, d_loss = model.train_step(x, y, loss=loss)
        g_epoch_loss += g_loss
        d_epoch_loss += d_loss

    g_epoch_loss /= len(loader)
    d_epoch_loss /= len(loader)

    return g_epoch_loss, d_epoch_loss


def min_sec(t):
    secs = int(t)
    mins = secs // 60
    secs = secs % 60
    return mins, secs


root_dir = '.'

os.system('wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz -P %s' % root_dir)
os.system('tar -zxf %s/facades.tar.gz -C %s' % (root_dir, root_dir))

root_dir = 'facades'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = Dataset(root_dir, transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)


model = Pix2pix().to(device)


N_EPOCHS = 350

for epoch in range(N_EPOCHS):
    print('EPOCH [{}]'.format(epoch + 1))

    start = time.time()
    g_train_loss, d_train_loss = train(train_loader, model, loss='gan')
    print('\t train time: {} min {} s, g_loss {:.3f}, d_loss {:.3f}'.format(
        *min_sec(time.time() - start), g_train_loss, d_train_loss
    ))


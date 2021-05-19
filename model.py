import torch
from torch import nn
import torch.nn.functional as F


class ContractionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        c = self.conv_block(x)
        p = self.maxpool(c)
        p = self.dropout(p)

        return c, p


class ExpansionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(dropout)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, c):
        u = self.upsample(x)
        u = self.dropout(u)
        u = torch.cat((u, c), 1)
        c = self.conv_block(u)

        return c


class UNet(nn.Module):
    def __init__(self, n_one_way_layers=5, d=64, dropout=0.1):
        super().__init__()

        self.n_one_way_layers = n_one_way_layers
        self.dropout = dropout

        self.n_cont_layers = n_one_way_layers
        self.n_exp_layers = n_one_way_layers

        cont_channels = [3] + [d * 2 ** i for i in range(self.n_cont_layers - 1)]
        self.cont_blocks = nn.ModuleList([
            ContractionBlock(in_c, out_c, self.dropout)
            for in_c, out_c in zip(cont_channels[:-1], cont_channels[1:])
        ])

        exp_channels = [d * 2 ** (i - 1) for i in range(self.n_exp_layers, 0, -1)]
        exp_channels.append(exp_channels[-1])
        self.exp_blocks = nn.ModuleList([
            ExpansionBlock(in_c, out_c, self.dropout)
            for in_c, out_c in zip(exp_channels[:-2], exp_channels[2:])
        ])

        mid_channels = [cont_channels[-1],
                        cont_channels[-1] // 2,
                        exp_channels[0] // 2]
        self.mid_block = nn.Sequential(
            nn.Conv2d(mid_channels[0], mid_channels[1], 3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[1]),
            nn.ReLU(),
            nn.Conv2d(mid_channels[1], mid_channels[2], 3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels[2]),
            nn.ReLU(),
        )

        self.conv1x1 = nn.Conv2d(exp_channels[-1], 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        shortcuts = []
        for i in range(len(self.cont_blocks)):
            c, x = self.cont_blocks[i](x)
            shortcuts.append(c)

        x = self.mid_block(x)

        for i in range(len(self.exp_blocks)):
            x = self.exp_blocks[i](x, shortcuts[-i - 1])

        x = self.tanh(self.conv1x1(x))

        return x


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv_bn(x)


class Discriminator(nn.Module):
    def __init__(self, d=32, n_layers=5):
        super().__init__()

        channels = [d * 2 ** i for i in range(n_layers - 1)] + [1]

        self.layers = nn.ModuleList([nn.Conv2d(6, d, 4, 2, 1)] + \
                                    [ConvBN(in_c, out_c) for in_c, out_c in zip(channels[:-1], channels[1:])])

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat((x, y), 1)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.leaky_relu(x)

        return self.sigmoid(x)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class Pix2pix(nn.Module):
    def __init__(self, alpha=100.0, max_lr=1e-3):
        super().__init__()

        self.generator = UNet()
        self.discriminator = Discriminator()

        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.alpha = alpha

        self.max_lr = max_lr
        self.g_opt = torch.optim.Adam(self.generator.parameters())
        self.d_opt = torch.optim.Adam(self.discriminator.parameters())

        self.step = 0

    def forward(self, x):
        return self.generator(x)

    def get_d_loss(self, x, y):
        d_fake_pred = self.discriminator(x, self.generator(x))
        d_fake_loss = self.mse(d_fake_pred, torch.zeros(d_fake_pred.shape).to(device))

        d_real_pred = self.discriminator(x, y)
        d_real_loss = self.mse(d_real_pred, torch.ones(d_real_pred.shape).to(device))

        return (d_real_loss + d_fake_loss) / 2

    def get_g_gan_loss(self, x, y):
        g_pred = self.generator(x)
        d_fake_pred = self.discriminator(x, g_pred)
        g_loss = torch.add(
            self.mse(d_fake_pred, torch.ones(d_fake_pred.shape).to(device)),
            self.alpha * self.l1_loss(g_pred, y)
        )
        return g_loss

    def train_step(self, x, y, loss='gan'):
        x, y = x.to(device), y.to(device)
        self.step += 1

        if loss == 'gan':
            self.update_lr(self.g_opt)
            self.update_lr(self.d_opt)

            # train generator
            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)
            self.g_opt.zero_grad()
            g_loss = self.get_g_gan_loss(x, y)
            g_loss.backward()
            self.g_opt.step()

            # train discriminator
            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)
            self.d_opt.zero_grad()
            d_loss = self.get_d_loss(x, y)
            d_loss.backward()
            self.d_opt.step()

            return g_loss, d_loss

        elif loss == 'l1':
            self.update_lr(self.g_opt)

            self.g_opt.zero_grad()
            g_loss = self.l1_loss(self.generator(x), y)
            g_loss.backward()
            self.g_opt.step()
            return g_loss, 0
        else:
            raise NotImplementedError('loss type "{}" is not found'.format(loss))

    def update_lr(self, optimizer, t_warmap=1000):
        for param_group in optimizer.param_groups:
            if self.step <= t_warmap:
                param_group['lr'] = self.step / t_warmap * self.max_lr
            else:
                param_group['lr'] = (t_warmap / self.step) ** 0.25 * self.max_lr
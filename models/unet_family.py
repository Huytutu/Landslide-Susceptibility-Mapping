"""U-Net family used in the paper: UNet, ResUNet, MUNet and ResMUNet.

All four variants share one encoder-decoder and differ only in two switches:

* ``residual``   -- add a (1x1-projected) identity shortcut to every conv block
* ``multiscale`` -- pass each skip connection through the Multi-Scale
  Connection (MSC) module before it is concatenated in the decoder

The attribute names are kept identical to the original per-model files so the
checkpoints in ``models/*_best_model.pth`` load unchanged.

The networks return **logits**. Apply ``torch.sigmoid`` exactly once, where
probabilities are needed (see ``losses.py`` and ``metrics.py``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Multiscale(nn.Module):
    """Multi-Scale Connection: four parallel branches, each with C/4 channels."""

    def __init__(self, in_channels):
        super().__init__()
        sub_channels = in_channels // 4
        # pixel-level (1x1)
        self.s0 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        # local context (3x3)
        self.s1 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        # wide context (3x3, dilation 3)
        self.s2 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        # salient features (3x3 max-pool, stride 1, then 1x1)
        self.s3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, sub_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat((self.s0(x), self.s1(x), self.s2(x), self.s3(x)), dim=1)


class ResidualConv(nn.Module):
    """Two 3x3 conv -> GroupNorm -> SiLU layers, optionally with a residual shortcut.

    With ``is_res=False`` this is the plain double-conv block of the UNet/MUNet
    baselines (note: GroupNorm + SiLU + dropout, not the original 2015 U-Net block).
    """

    def __init__(self, in_channels, out_channels, droprate, is_res=False):
        super().__init__()
        self.is_res = is_res

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout2d(droprate)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()

        if is_res:
            self.resid_layer = nn.Identity() if in_channels == out_channels else \
                               nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.drop1(self.act1(self.norm1(self.conv1(x))))
        out = self.act2(self.norm2(self.conv2(out)))
        if self.is_res:
            out = (out + self.resid_layer(x)) / 1.414
        return out


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, is_res):
        super().__init__()
        self.conv = ResidualConv(in_channels, out_channels, dropout_rate, is_res)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x


class Upsample(nn.Module):
    """Bilinear x2 upsampling, concat with the (optionally MSC-enriched) skip, then a conv block.

    ``in_channels`` is the channel count after concatenation; half of it comes
    from the skip connection.
    """

    def __init__(self, in_channels, out_channels, dropout_rate, is_res, multiscale):
        super().__init__()
        if multiscale:
            self.ms = Multiscale(in_channels // 2)
        self.layer = ResidualConv(in_channels, out_channels, dropout_rate, is_res)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if hasattr(self, 'ms'):
            skip = self.ms(skip)
        return self.layer(torch.cat((x, skip), dim=1))


class UNetFamily(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, residual=False, multiscale=False, name=None):
        super().__init__()
        self.in_channels = in_channels
        self.name = name or ('Res' if residual else '') + ('MUNet' if multiscale else 'UNet')

        # encoder
        self.down1 = Downsample(in_channels, 64, 0.1, residual)  # 128x128x6 -> 64x64x64
        self.down2 = Downsample(64, 128, 0.1, residual)          # -> 32x32x128
        self.down3 = Downsample(128, 256, 0.2, residual)         # -> 16x16x256
        self.down4 = Downsample(256, 512, 0.2, residual)         # -> 8x8x512

        # bottleneck
        self.bottle = ResidualConv(512, 512, 0.3, residual)      # 8x8x512 -> 8x8x512

        # decoder
        self.up1 = Upsample(512 * 2, 256, 0.2, residual, multiscale)
        self.up2 = Upsample(256 * 2, 128, 0.2, residual, multiscale)
        self.up3 = Upsample(128 * 2, 64, 0.1, residual, multiscale)
        self.up4 = Upsample(64 * 2, 32, 0.1, residual, multiscale)

        # output (logits)
        self.out_layer = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1_pool, x1 = self.down1(x)
        x2_pool, x2 = self.down2(x1_pool)
        x3_pool, x3 = self.down3(x2_pool)
        x4_pool, x4 = self.down4(x3_pool)

        bottle = self.bottle(x4_pool)

        d1 = self.up1(bottle, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        return self.out_layer(d4)

    def __str__(self):
        return self.name


def UNET(in_channels=6, out_channels=1):
    return UNetFamily(in_channels, out_channels, residual=False, multiscale=False, name='UNet')


def RESUNET(in_channels=6, out_channels=1):
    return UNetFamily(in_channels, out_channels, residual=True, multiscale=False, name='ResUNet')


def MUNET(in_channels=6, out_channels=1):
    return UNetFamily(in_channels, out_channels, residual=False, multiscale=True, name='MUNet')


def ResMUNET(in_channels=6, out_channels=1):
    return UNetFamily(in_channels, out_channels, residual=True, multiscale=True, name='ResMUNet')


MODELS = {'UNet': UNET, 'ResUNet': RESUNET, 'MUNet': MUNET, 'ResMUNet': ResMUNET}


def build_model(name, in_channels=6, out_channels=1):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")
    return MODELS[name](in_channels, out_channels)


def default_checkpoint(name):
    """Path of the released checkpoint for ``name`` (relative to the repo root)."""
    return f'models/{name}_best_model.pth'

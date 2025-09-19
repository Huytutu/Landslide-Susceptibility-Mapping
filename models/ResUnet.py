import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, droprate, is_res=True):
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
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        if self.is_res:
            out = (out + self.resid_layer(x)) / 1.414
        return out

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.conv = ResidualConv(in_channels, out_channels, dropout_rate)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super().__init__()
        self.layer = ResidualConv(in_channels, out_channels, dropout_rate)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat((x, skip), dim=1)
        return self.layer(x)

class RESUNET(nn.Module):
    def __init__(self,in_channels=6,out_channels=1):
        super(RESUNET,self).__init__()
        self.in_channels = in_channels

        # encoder
        self.down1 = Downsample(in_channels,64,0.1) # 128x128x6 -> 64x64x64
        self.down2 = Downsample(64,128,0.1) # 64x64x64 -> 32x32x128
        self.down3 = Downsample(128,256,0.2) # 32x32x128 -> 16x16x256
        self.down4 = Downsample(256,512,0.2) # 16x16x256 -> 8x8x512

        # bottleneck
        self.bottle = ResidualConv(512,512,0.3) # 8x8x512 -> 8x8x512

        # decoder
        self.up1 = Upsample(512*2,256,0.2)
        self.up2 = Upsample(256*2,128,0.2)
        self.up3 = Upsample(128*2,64,0.1)
        self.up4 = Upsample(64*2,32,0.1)

        # output
        self.out_layer = nn.Conv2d(32,out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        x1_pool, x1 = self.down1(x)
        x2_pool, x2 = self.down2(x1_pool)
        x3_pool, x3 = self.down3(x2_pool)
        x4_pool, x4 = self.down4(x3_pool)

        bottle = self.bottle(x4_pool)

        d1 = self.up1(bottle,x4)
        d2 = self.up2(d1,x3)
        d3 = self.up3(d2,x2)
        d4 = self.up4(d3,x1)

        out = torch.sigmoid(self.out_layer(d4))
        return out
    
    def __str__(self):
        return 'ResUNet'
import torch
import torch.nn as nn

"""
    Double convolution
    Conv2D (same in and out size) -> Batch Normalization -> ReLU 
"""
class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

"""
    Downsample
    Divide H and W by 2 and then call DoubleConv
"""
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

"""
    Upsample
    Add info learned from 2*2 on pixel then cat those info with the skip sotcked from down
"""
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        #we divide in_ch by 2 due to keep a stable channel dimension with the cat operation incoming
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x=torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    pass
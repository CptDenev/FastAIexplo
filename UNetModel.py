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
    
"""
class Up(nn.Module):
    pass


class UNet(nn.Module):
    pass
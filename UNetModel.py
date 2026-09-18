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
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1) #add dropout due to too stable loss on val set
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

"""
    UNet global
"""
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, base_filters=64):
        super().__init__()
        f = base_filters
        #ENCODERS
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f, f*2)
        self.enc3 = Down(f*2, f*4)
        self.enc4 = Down(f*4, f*8)
        #BOTTLENECK
        self.bottleneck_pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f*8, f*16)
        #DECODERS
        self.dec4 = Up(f*16, f*8)
        self.dec3 = Up(f*8, f*4)
        self.dec2 = Up(f*4, f*2)
        self.dec1 = Up(f*2, f)
        #HEAD Conv 1*1 -> num_classes by pixel
        self.head = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        #encoders
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        #bottleneck
        b = self.bottleneck(self.bottleneck_pool(e4))
        #decoders
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)        
        #head
        return self.head(d1)
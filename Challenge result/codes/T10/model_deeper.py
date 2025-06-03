from torch.nn import functional as F
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class Unet(nn.Module):
    """
    Basic U-net model
    """

    def __init__(self, input_size, output_size):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_size,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
                               
        self.se1 = SELayer(32)

        self.pool1 = nn.Conv3d(in_channels=32,
                               out_channels=32,
                               kernel_size=2,
                               stride=2)

        self.dp1 = nn.Dropout3d(p=0.25)

        self.conv2 = nn.Conv3d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)

        self.se2 = SELayer(64)

        self.pool2 = nn.Conv3d(in_channels=64,
                               out_channels=64,
                               kernel_size=2,
                               stride=2)
        self.dp2 = nn.Dropout3d(p=0.5)

        self.conv3 = nn.Conv3d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)

        self.se3 = SELayer(128)

        self.pool3 = nn.Conv3d(in_channels=128,
                               out_channels=128,
                               kernel_size=2,
                               stride=2)
        self.dp3 = nn.Dropout3d(p=0.5)

        self.conv4 = nn.Conv3d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               padding=1)

        self.se4 = SELayer(256)
        self.dp4 = nn.Dropout3d(p=0.5)
        self.pool4 = nn.Conv3d(in_channels=256,
                               out_channels=256,
                               kernel_size=2,
                               stride=2)
        
        self.conv5 = nn.Conv3d(in_channels=256,
                               out_channels=512,
                               kernel_size=3,
                               padding=1)

        self.se5 = SELayer(512)

        self.up1 = nn.ConvTranspose3d(in_channels=512,
                                      out_channels=256,
                                      kernel_size=2,
                                      stride=2)

        self.conv6 = nn.Conv3d(in_channels=256,
                               out_channels=256,
                               kernel_size=3,
                               padding=1)
        self.dp6 = nn.Dropout3d(p=0.5)

        self.up2 = nn.ConvTranspose3d(in_channels=256,
                                      out_channels=128,
                                      kernel_size=2,
                                      stride=2)
        self.conv7 = nn.Conv3d(in_channels=128,
                               out_channels=128,
                               kernel_size=3,
                               padding=1)
        self.dp7 = nn.Dropout3d(p=0.5)

        self.up3 = nn.ConvTranspose3d(in_channels=128,
                                      out_channels=64,
                                      kernel_size=2,
                                      stride=2)
        self.conv8 = nn.Conv3d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               padding=1)
        self.dp8 = nn.Dropout3d(p=0.5)

        self.up4 = nn.ConvTranspose3d(in_channels=64,
                                      out_channels=32,
                                      kernel_size=2,
                                      stride=2)
        self.conv9 = nn.Conv3d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)

        self.conv10 = nn.Conv3d(in_channels=32,
                               out_channels=output_size,
                               kernel_size=1)

    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        s1 = self.se1(x1)
        x1p = self.dp1(self.pool1(x1))
        x2 = F.relu(self.conv2(x1p))
        s2 = self.se2(x2)
        x2p = self.dp2(self.pool2(x2))
        x3 = F.relu(self.conv3(x2p))
        s3 = self.se3(x3)
        x3p = self.dp3(self.pool3(x3))
        x4 = F.relu(self.conv4(x3p))
        s4 = self.se4(x4)
        x4p = self.dp4(self.pool4(x4))

        x5 = F.relu(self.se5(self.conv5(x4p)))

        up1 = self.up1(x5)
        x6 = F.relu(self.dp6(self.conv6(up1+s4))) 
        up2 = self.up2(x6)
        x7 = F.relu(self.dp7(self.conv7(up2+s3)))
        up3 = self.up3(x7)
        x8 = F.relu(self.conv8(self.dp8((up3+s2))))
        up4 = self.up4(x8)
        x9 = F.relu(self.conv9((up4+s1)))

        out = self.conv10(x9)
        return out
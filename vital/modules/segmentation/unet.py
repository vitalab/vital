import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    '''
    Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation
    Link - https://arxiv.org/abs/1505.04597

    Parameters:
    input_channels (int) - Number of input channels
    output_channels (int) - Number of input channels
    bilinear (bool) - Whether to use bilinear interpolation or transposed
    convolutions for upsampling.
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 init_channels: int = 32,
                 bilinear: bool = False,
                 dropout_prob: float = 0.0,
                 use_batchnorm: bool = True):

        super().__init__()
        self.layer1 = DoubleConv(in_channels, init_channels // 2, dropout_prob / 2, use_batchnorm)
        self.layer2 = Down(init_channels // 2, init_channels, dropout_prob, use_batchnorm)
        self.layer3 = Down(init_channels, init_channels * 2, dropout_prob, use_batchnorm)
        self.layer4 = Down(init_channels * 2, init_channels * 4, dropout_prob, use_batchnorm)
        self.layer5 = Down(init_channels * 4, init_channels * 8, dropout_prob, use_batchnorm)
        self.layer6 = Down(init_channels * 8, init_channels * 16, dropout_prob, use_batchnorm)

        self.layer7 = Up(init_channels * 16, init_channels * 8, dropout_prob, use_batchnorm, bilinear=bilinear)
        self.layer8 = Up(init_channels * 8, init_channels * 4, dropout_prob, use_batchnorm, bilinear=bilinear)
        self.layer9 = Up(init_channels * 4, init_channels * 2, dropout_prob, use_batchnorm, bilinear=bilinear)
        self.layer10 = Up(init_channels * 2, init_channels, dropout_prob, use_batchnorm, bilinear=bilinear)
        self.layer11 = Up(init_channels, init_channels // 2, 0, use_batchnorm, bilinear=bilinear)

        self.layer12 = nn.Conv2d(init_channels // 2, out_channels, kernel_size=1)

        # Use Xavier initialisation for weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)

        out = self.layer7(x6, x5)
        out = self.layer8(out, x4)
        out = self.layer9(out, x3)
        out = self.layer10(out, x2)
        out = self.layer11(out, x1)

        return self.layer12(out)


class DoubleConv(nn.Module):
    """
    Double Convolution and BN and ReLU
    (3x3 conv -> BN -> ReLU) ** 2
    """

    def __init__(self, in_ch, out_ch, dropout_prob, use_batchnorm):
        super().__init__()
        if use_batchnorm:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Combination of MaxPool2d and DoubleConv in series
    """

    def __init__(self, in_ch, out_ch, dropout_prob, use_bn):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch, dropout_prob, use_bn)
        )

    def forward(self, x, dropout_prob=0):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by double 3x3 convolution.
    """

    def __init__(self, in_ch, out_ch, dropout_prob, use_bn, bilinear=False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch, dropout_prob, use_bn)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


if __name__ == '__main__':
    from torchsummary import summary

    model = UNet(in_channels=1, out_channels=4)

    summary(model, (1, 256, 256), device='cpu')

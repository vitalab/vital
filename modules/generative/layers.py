from torch import nn


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)


def conv_transpose3x3(in_channels, out_channels, stride=1, padding=1, output_padding=0):
    """2x2 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding,
                              output_padding=output_padding)

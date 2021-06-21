from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 affine,
                 track_running_stats=True):
        super().__init__()

        self.op = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                dilation, 
                bias=False
            ),
            nn.BatchNorm2d(
                out_channels,
                affine=affine,
                track_running_stats=track_running_stats
            ),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)

class Conv1x1BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, affine=True):
        super().__init__()

        self.conv1x1 = ConvBNReLU(in_channels, out_channels, 1, 1, 0, 1, affine)

    def forward(self, x):
        return self.conv1x1(x)

class Conv3x3BNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, affine=True):
        super().__init__()

        self.conv3x3 = ConvBNReLU(in_channels, out_channels, 3, stride, 1, 1, affine)

    def forward(self, x):
        return self.conv3x3(x)

class ResidualBLock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 stride=1,
                 affine=True):
        super().__init__()

        assert stride == 1 or stride == 2, 'invalid stride {}'.format(stride)

        self.conv1 = Conv3x3BNReLU(in_channels, out_channels, stride, affine)

        self.conv2 = Conv3x3BNReLU(out_channels, out_channels, affine)

        self.downsample = nn.Identity()
        if stride == 2:
            ## Pooling + Conv1x1
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    bias=False
                )
            )
        elif in_channels != out_channels:
            ## Conv1x1
            self.downsample = Conv1x1BNReLU(in_channels, out_channels, affine)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        residual = self.downsample(x)

        out += residual
        return out




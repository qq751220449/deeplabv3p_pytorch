import torch.nn as nn
import torch
import math


class conv_depth(nn.Module):        # Depthwise卷积
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, bias=False):
        super(conv_depth, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=((kernel_size-1)*(dilation-1)+kernel_size)//2,
            groups=in_ch,
            bias=bias,
            dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class conv_point(nn.Module):        # Pointwise卷积
    def __init__(self, in_ch, out_ch, bias=False):
        super(conv_point, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
            dilation=1,
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class seq_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, bias=False):
        super(seq_conv, self).__init__()
        self.seq_conv = nn.Sequential(
            conv_depth(in_ch=in_ch, out_ch=in_ch, kernel_size=3, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(in_ch),
            conv_point(in_ch=in_ch, out_ch=out_ch, bias=bias),
        )

    def forward(self, x):
        out = self.seq_conv(x)
        return out


class seq_conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super(seq_conv_bn, self).__init__()
        self.seq_conv_bn = nn.Sequential(
            seq_conv(in_ch=in_ch, out_ch=out_ch, stride=stride, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.seq_conv_bn(x)
        return out



class seq_conv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super(seq_conv_bn_relu, self).__init__()
        self.seq_conv_bn_relu = nn.Sequential(
            seq_conv(in_ch=in_ch, out_ch=out_ch, stride=stride, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.seq_conv_bn_relu(x)
        return out


class conv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1):
        super(conv_bn_relu, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=((kernel_size-1)*(dilation-1)+kernel_size)//2, stride=stride, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv_bn_relu(x)
        return out


class xception_block(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, dilation=1):
        super(xception_block, self).__init__()
        if isinstance(out_ch, int):
            if stride != 1 or in_ch != out_ch:
                self.skip = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, bias=False, dilation=dilation),
                    nn.BatchNorm2d(out_ch)
                )
            else:
                self.skip = nn.Sequential()

            self.seq_conv1 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=in_ch, out_ch=out_ch, stride=1, dilation=dilation),
            )
            self.seq_conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=out_ch, out_ch=out_ch, stride=1, dilation=dilation),
            )
            if stride != 1:
                self.seq_conv3 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=out_ch, out_ch=out_ch, stride=stride, dilation=dilation),
            )
            else:
                self.seq_conv3 = nn.Sequential(
                    nn.ReLU(inplace=True),
                    seq_conv_bn(in_ch=out_ch, out_ch=out_ch, stride=1, dilation=dilation),
                )
        else:
            if stride != 1 or in_ch != out_ch:
                self.skip = nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch[2], kernel_size=1, stride=stride, bias=False, dilation=dilation),
                    nn.BatchNorm2d(out_ch[2])
                )
            else:
                self.skip = nn.Sequential()

            self.seq_conv1 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=in_ch, out_ch=out_ch[0], stride=1, dilation=dilation),
            )
            self.seq_conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=out_ch[0], out_ch=out_ch[1], stride=1, dilation=dilation),
            )
            if stride != 1:
                self.seq_conv3 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=out_ch[1], out_ch=out_ch[2], stride=stride, dilation=dilation),
            )
            else:
                self.seq_conv3 = nn.Sequential(
                    nn.ReLU(inplace=True),
                    seq_conv_bn(in_ch=out_ch[1], out_ch=out_ch[2], stride=1, dilation=dilation),
                )

    def forward(self, input):
        skip = self.skip(input)
        x = self.seq_conv1(input)
        x = self.seq_conv2(x)
        x = self.seq_conv3(x)
        x = x + skip
        return x


class EntryFlow(nn.Module):
    def __init__(self, output_stride):
        super(EntryFlow, self).__init__()
        self.model_block1 = nn.Sequential(
            # conv1
            conv_bn_relu(in_ch=3, out_ch=32, kernel_size=3, stride=2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            # block1
            xception_block(in_ch=64, out_ch=128, stride=2),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.model_block2 = nn.Sequential(
            # block2
            xception_block(in_ch=128, out_ch=256, stride=2),
        )
        if output_stride == 16:
            self.block3 = nn.Sequential(
                xception_block(in_ch=256, out_ch=728, stride=2),
            )
        elif output_stride == 8:
            self.block3 = nn.Sequential(
                xception_block(in_ch=256, out_ch=728, stride=1),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.model_block1(x)
        low_level_feat = self.relu(x)
        x = self.model_block2(x)
        x = self.block3(x)
        return x, low_level_feat


class MiddleFlow(nn.Module):
    def __init__(self,output_stride):
        super(MiddleFlow, self).__init__()
        rep = []
        for index in range(16):
            if output_stride == 16:
                rep.append(xception_block(in_ch=728, out_ch=728, stride=1, dilation=1))
            elif output_stride == 8:
                rep.append(xception_block(in_ch=728, out_ch=728, stride=1, dilation=2))
        self.model = nn.Sequential(
                *rep
            )

    def forward(self, x):
        x = self.model(x)
        return x


class ExitFlow(nn.Module):
    def __init__(self, output_stride):
        super(ExitFlow, self).__init__()
        if output_stride == 16:
            self.skip = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=728, out_channels=1024, kernel_size=1, stride=1, bias=False, dilation=2),
                nn.BatchNorm2d(1024)
            )

            self.seq_conv1 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=728, out_ch=728, stride=1, dilation=1),
            )
            self.seq_conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=728, out_ch=1024, stride=1, dilation=1),
            )
            self.seq_conv3 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=1024, out_ch=1024, stride=1, dilation=2),
            )

            self.conv3 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=1024, out_ch=1536, stride=1, dilation=2),
            )
            self.conv4 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=1536, out_ch=1536, stride=1, dilation=2),
            )
            self.conv5 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=1536, out_ch=2048, stride=1, dilation=2),
                nn.ReLU(inplace=True)
            )
        elif output_stride == 8:
            self.skip = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=728, out_channels=1024, kernel_size=1, stride=1, bias=False, dilation=4),
                nn.BatchNorm2d(1024)
            )

            self.seq_conv1 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=728, out_ch=728, stride=1, dilation=2),
            )
            self.seq_conv2 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=728, out_ch=1024, stride=1, dilation=2),
            )
            self.seq_conv3 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=1024, out_ch=1024, stride=1, dilation=4),
            )

            self.conv3 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=1024, out_ch=1536, stride=1, dilation=4),
            )
            self.conv4 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=1536, out_ch=1536, stride=1, dilation=4),
            )
            self.conv5 = nn.Sequential(
                nn.ReLU(inplace=True),
                seq_conv_bn(in_ch=1536, out_ch=2048, stride=1, dilation=4),
                nn.ReLU(inplace=True)
            )

    def forward(self, input):
        skip = self.skip(input)
        x = self.seq_conv1(input)
        x = self.seq_conv2(x)
        x = self.seq_conv3(x)
        x_out = skip + x
        x_out = self.conv3(x_out)
        x_out = self.conv4(x_out)
        x_out = self.conv5(x_out)
        return x_out


class Xception(nn.Module):
    def __init__(self, output_stride):
        super(Xception, self).__init__()
        self.entry_flow = nn.Sequential(
            EntryFlow(output_stride),
        )
        self.middle_flow = nn.Sequential(
            MiddleFlow(output_stride),
        )
        self.exit_flow = nn.Sequential(
            ExitFlow(output_stride),
        )
        self._init_weight()

    def forward(self, x):
        x, low_level_feat = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# https://github.com/huxycn/deeplabv3plus-pytorch/blob/master/modeling/backbone/xception.py
# https://github.com/YudeWang/deeplabv3plus-pytorch/blob/master/lib/net/xception.py


if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conv = Xception(16).to(device)
    print(conv)
    print(summary(conv, input_size=(3, 1024, 384), batch_size=10))

    input_data = torch.rand(1, 3, 1024, 384).to(device)
    output_data, low = conv(input_data)
    print(output_data.size())
    print(low.size())

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.config import encoder_channel

class _ASPPModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super(_ASPPModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)*(dilation-1)+kernel_size)//2, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self._init_weight()

    def forward(self, x):
        x = self.model(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backone, output_stride):
        super(ASPP, self).__init__()

        if backone == "xception":
            in_ch = 2048
        elif backone == "resnet":
            in_ch = 2048
        else:
            raise NotImplementedError

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(in_ch=in_ch, out_ch=encoder_channel, kernel_size=1, dilation=dilations[0])
        self.aspp2 = _ASPPModule(in_ch=in_ch, out_ch=encoder_channel, kernel_size=3, dilation=dilations[1])
        self.aspp3 = _ASPPModule(in_ch=in_ch, out_ch=encoder_channel, kernel_size=3, dilation=dilations[2])
        self.aspp4 = _ASPPModule(in_ch=in_ch, out_ch=encoder_channel, kernel_size=3, dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_ch, out_channels=encoder_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(encoder_channel),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=encoder_channel*5, out_channels=encoder_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(encoder_channel),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride):
    return ASPP(backbone, output_stride)


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_aspp("xception", output_stride=8).to(device)
    print(summary(model, input_size=(2048, 128, 48), batch_size=1))


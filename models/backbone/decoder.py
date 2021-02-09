import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.config import decoder_channel


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone):
        super(Decoder, self).__init__()
        if backbone == "xception":
            low_level_in_ch = 128
        elif backbone == "resnet":
            low_level_in_ch = 256
        else:
            raise NotImplementedError
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=low_level_in_ch, out_channels=decoder_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channel),
            nn.ReLU(inplace=True)
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(decoder_channel+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone):
    return Decoder(num_classes, backbone)
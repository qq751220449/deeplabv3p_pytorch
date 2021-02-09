import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.aspp import build_aspp
from models.backbone.decoder import build_decoder
from models.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=9, freeze_bn=False):
        super(DeepLab, self).__init__()

        self.backbone = build_backbone(backbone, output_stride)
        self.aspp = build_aspp(backbone, output_stride)
        self.decoder = build_decoder(num_classes, backbone)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = DeepLab(backbone='xception', output_stride=16).to(device)         # xception

    print(model)
    print(summary(model, input_size=(3, 1024, 384), batch_size=5))

    model.eval()
    input = torch.rand(1, 3, 1024, 384).to(device)
    output = model(input)
    print(output.size())

# https://www.cnblogs.com/wanghui-garcia/p/10895397.html
# https://github.com/huxycn/deeplabv3plus-pytorch/blob/master/utils/lr_scheduler.py
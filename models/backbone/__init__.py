from models.backbone.xception import Xception
from models.backbone.atros_resnet import ResNet101


def build_backbone(backbone="xception", output_stride=16):
    if backbone == "xception":
        model = Xception(output_stride=output_stride)
        return model
    elif backbone == "resnet":
        model = ResNet101(output_stride=output_stride)
        return model
    else:
        raise NotImplementedError
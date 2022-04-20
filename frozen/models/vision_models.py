from timm.models.helpers import build_model_with_cfg
from timm.models.nfnet import NormFreeNet, default_cfgs as nfnet_dcfgs, model_cfgs as nfnet_cfgs
from timm.models.resnet import ResNet, default_cfgs as resnet_dcfgs
from torch import nn

from utils import freeze_module


class NormFreeNetModel(NormFreeNet):
    def freeze(self, num_frozen_stages):
        freeze_module(self.stem)
        for stage in self.stages[:num_frozen_stages]:
            freeze_module(stage)


class ResNetModel(ResNet):
    def freeze(self, num_frozen_stages):
        freeze_module(self.conv1)
        freeze_module(self.bn1)
        stages = [getattr(self, f'layer{i+1}') for i in range(num_frozen_stages)]
        for stage in stages:
            freeze_module(stage)


class PretrainedVisionEncoder(nn.Module):
    CONFIGS = dict(
        nf_resnet50=dict(
            default_cfg=nfnet_dcfgs['nf_resnet50'],
            model_cfg=nfnet_cfgs['nf_resnet50'],
            feature_cfg=dict(flatten_sequential=True)
        ),
        resnet18=dict(
            default_cfg=resnet_dcfgs['resnet18']
        ),
        resnet34=dict(
            default_cfg=resnet_dcfgs['resnet34']
        ),
        resnet50=dict(
            default_cfg=resnet_dcfgs['resnet50']
        )
    )
    CLASSES = dict(
        nf_resnet50=NormFreeNetModel,
        resnet50=ResNetModel
    )

    @classmethod
    def from_pretrained(cls, key, pretrained=True):
        kwargs = cls.CONFIGS[key]
        return build_model_with_cfg(cls.CLASSES[key], key, pretrained=pretrained, **kwargs)




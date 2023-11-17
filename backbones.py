# import torch
# import torch.nn as nn

# from torch import Tensor
# from typing import Type
# from torchvision.models import wide_resnet50_2
# class BasicBlock(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  stride: int = 1,
#                  expansion: int = 1,
#                  downsample: Type[nn.Module] = None) -> None:
#         super(BasicBlock, self).__init__()
#         self.expansion = expansion
#         self.downsample = downsample
#         self.conv1 = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(
#             out_channels,
#             out_channels * self.expansion,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
    
#     def forward(self, x: Tensor):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         out += identity
#         out = self.relu(out)
#         return out

# class ResNet18(nn.Module):
#     def __init__(self,
#                  img_channels = 1,
#                  num_layers = 18,
#                  block = BasicBlock,
#                  ):
#         super(ResNet18, self).__init__()
#         if num_layers == 18:
#             layers = [2, 2, 2, 2]
#             self.expansion = 1
        
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(
#             img_channels,
#             self.in_channels,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#     def _make_layer(self,
#                     block: Type[nn.Module],
#                     out_channels: int,
#                     blocks: int,
#                     stride: int = 1) -> nn.Sequential:
#         downsample = None
#         if stride != 1:
#             downsample = nn.Sequential(
#                 nn.Conv2d(
#                     self.in_channels,
#                     out_channels * self.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False),
#                 nn.BatchNorm2d(out_channels * self.expansion),
#             )
#         layers = []
#         layers.append(
#             block(self.in_channels, out_channels, stride, self.expansion,
#                   downsample)
#                   )
#         self.in_channels = out_channels * self.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
#         return nn.Sequential(*layers)
    
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         return x

# if __name__ == '__main__':
#     model = ResNet18()
#     print(model.layer3)

import timm  # noqa
import torch
import torchvision.models as models  # noqa

def load_ref_wrn50():
    
    import resnet 
    return resnet.wide_resnet50_2(True)

_BACKBONES = {
    "cait_s24_224" : "cait.cait_S24_224(True)",
    "cait_xs24": "cait.cait_XS24(True)",
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    "resnet18": "models.resnet18(pretrained=True)",
    "resnet50": "models.resnet50(pretrained=True)",
    "mc3_resnet50": "load_mc3_rn50()", 
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "ref_wideresnet50": "load_ref_wrn50()",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}


def load(name):
    return eval(_BACKBONES[name])
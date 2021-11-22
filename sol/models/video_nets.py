import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(self, original_resnet):
        super(Resnet, self).__init__()
        self.features = nn.Sequential(*list(original_resnet.children())[:-2])
        self.feat_dim = original_resnet.fc.in_features

    def get_embed_dim(self):
        return self.feat_dim

    def forward(self, x):

        x = self.features(x)

        return x


class VGG16(nn.Module):
    def __init__(self, original_vgg16, init_weights=False):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(*list(original_vgg16.children())[:-2])

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        _, C, H, W = x.size()
        x = x.view(B, T, C, H, W)
        return x

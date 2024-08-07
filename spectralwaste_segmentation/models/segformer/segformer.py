import torch

from .mix_transformer import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .segformer_head import SegFormerHead

from torch.utils import model_zoo

class SegFormer(torch.nn.Module):
    def __init__(self, in_chans, num_classes, backbone):
        super().__init__()

        if backbone == 'mit_b0':
            self.backbone = mit_b0(in_chans=in_chans)
            self.decode_head = SegFormerHead(
                in_channels=[32, 64, 160, 256],
                num_classes = num_classes,
                embedding_dim=256,
                dropout_ratio=0.1
           )
        elif backbone == 'mit_b1':
            self.backbone = mit_b1()
            self.decode_head = SegFormerHead(
                in_channels=[64, 128, 320, 512],
                num_classes = num_classes,
                embedding_dim=768,
                dropout_ratio=0.1
           )
        elif backbone == 'mit_b2':
            self.backbone = mit_b2()
            self.decode_head = SegFormerHead(
                in_channels=[64, 128, 320, 512],
                num_classes = num_classes,
                embedding_dim=768,
                dropout_ratio=0.1
           )
        else:
            raise NotImplemented('Incorrect backbone')

    def forward(self, x):
        out = self.backbone(x)
        out = self.decode_head(out)
        out = torch.nn.functional.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return out

import torch
from .dual_segformer import mit_b5, mit_b4, mit_b2, mit_b0
from .mlp_decoder import DecoderHead


class EncoderDecoder(torch.nn.Module):
    def __init__(self, extra_in_chans=3, num_classes=5, encoder='mit_b5', cfg=None):
        super().__init__()

        self.norm_layer = torch.nn.BatchNorm2d

        if encoder == 'mit_b0':
            self.channels = [32, 64, 160, 256]
            self.backbone = mit_b0(extra_in_chans=extra_in_chans, norm_fuse=self.norm_layer)
        elif encoder == 'mit_b5':
            self.channels = [64, 128, 320, 512]
            self.backbone = mit_b5(extra_in_chans=extra_in_chans, norm_fuse=self.norm_layer)

        self.decoder = DecoderHead(in_channels=self.channels, num_classes=num_classes, norm_layer=self.norm_layer, embed_dim=256)

    def forward(self, rgb, modal_x):
        # Encode images with backbone and decode into a semantic segmentation
        # map of the same size as input

        orig_size = rgb.shape

        x = self.backbone(rgb, modal_x)
        out = self.decoder(x)
        out = torch.nn.functional.interpolate(out, size=orig_size[2:], mode='bilinear', align_corners=False)

        return out
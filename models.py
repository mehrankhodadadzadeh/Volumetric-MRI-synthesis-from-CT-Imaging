# models.py ─────────────────────────────────────────────────────────
"""
Multiple 3-D generators + a factory.
Add new models simply by registering them in GENERATOR_REGISTRY.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR, AttentionUnet as MONAI_AttentionUnet


# ------------------------------------------------------------------ #
# 1.  Common building block
# ------------------------------------------------------------------ #
class DoubleConv3D(nn.Module):
    """(Conv ➜ BN ➜ LeakyReLU ➜ Dropout) × 2"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),

            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------------ #
# 2.  Plain 3-D U-Net (identity output + dropout)
# ------------------------------------------------------------------ #
class UNet3D(nn.Module):
    """
    3-D U-Net with Monte-Carlo Dropout support (identity output).
    """
    def __init__(
        self,
        in_channels:  int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        C = base_channels
        dr = dropout_rate                     # shorthand

        self.enc1  = DoubleConv3D(in_channels, C,     dr)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2  = DoubleConv3D(C, 2 * C,   dr)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3  = DoubleConv3D(2 * C, 4 * C, dr)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(4 * C, 8 * C, dr)

        self.up3  = nn.ConvTranspose3d(8 * C, 4 * C, 2, stride=2)
        self.dec3 = DoubleConv3D(8 * C, 4 * C, dr)

        self.up2  = nn.ConvTranspose3d(4 * C, 2 * C, 2, stride=2)
        self.dec2 = DoubleConv3D(4 * C, 2 * C, dr)

        self.up1  = nn.ConvTranspose3d(2 * C, C, 2, stride=2)
        self.dec1 = DoubleConv3D(2 * C, C, dr)

        # identity output
        self.final = nn.Conv3d(C, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b  = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


# ------------------------------------------------------------------ #
# 3.  Patch discriminator (unchanged)
# ------------------------------------------------------------------ #
class Discriminator3D(nn.Module):
    """Lightweight patch discriminator without dropout."""
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = DoubleConv3D(in_channels, 32);  self.pool1 = nn.AvgPool3d(2)
        self.conv2 = DoubleConv3D(32, 64);           self.pool2 = nn.AvgPool3d(2)
        self.conv3 = DoubleConv3D(64, 128);          self.pool3 = nn.AvgPool3d(2)
        self.conv4 = DoubleConv3D(128, 256)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc  = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)
        return self.fc(self.gap(x).flatten(1))


# ------------------------------------------------------------------ #
# 4.  Alternative generators with dropout=0.25
# ------------------------------------------------------------------ #
class SwinUNETRGenerator(SwinUNETR):
    """MONAI Swin-UNETR with dropout 0.25 and identity output."""
    def __init__(self, img_size, in_channels=1, out_channels=1, **kw):
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            drop_rate=0.25,            # 🛡️ added
            attn_drop_rate=0.25,        # 🛡️ optional but good
            **kw,
        )

    def forward(self, x):
        return super().forward(x)


class AttentionUNet3D_MONAI(nn.Module):
    """MONAI 3D Attention U-Net with dropout 0.25 and identity output."""
    def __init__(self, in_channels=1, out_channels=1, **kw):
        super().__init__()
        self.net = MONAI_AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels = (64, 128, 256, 512, 1024),
            strides  = (2, 2, 2, 2),
            dropout=0.25,               # 🛡️ added
            **kw,
        )

    def forward(self, x):
        return self.net(x)

class Discriminator3D_deeper(nn.Module):
    """Moderately strengthened 3D patch discriminator (5 conv blocks)."""
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = DoubleConv3D(in_channels, 32);  self.pool1 = nn.AvgPool3d(2)
        self.conv2 = DoubleConv3D(32, 64);           self.pool2 = nn.AvgPool3d(2)
        self.conv3 = DoubleConv3D(64, 128);          self.pool3 = nn.AvgPool3d(2)
        self.conv4 = DoubleConv3D(128, 256)
        self.conv5 = DoubleConv3D(256, 512)          # ✅ added layer
        self.gap   = nn.AdaptiveAvgPool3d(1)
        self.fc    = nn.Linear(512, 1)               # ✅ updated

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)                            # ✅ extra block
        return self.fc(self.gap(x).flatten(1))


# ------------------------------------------------------------------ #
# 5.  Registry + factory
# ------------------------------------------------------------------ #
GENERATOR_REGISTRY = {
    "unet3d":         UNet3D,
    "swin_unetr":     SwinUNETRGenerator,
    "attention_unet": AttentionUNet3D_MONAI,
}

def build_generator(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in GENERATOR_REGISTRY:
        raise ValueError(
            f"Unknown generator '{name}'. "
            f"Available: {list(GENERATOR_REGISTRY)}"
        )
    return GENERATOR_REGISTRY[key](**kwargs)

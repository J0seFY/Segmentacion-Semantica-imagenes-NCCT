"""
Model Definitions for Ischemic Stroke Segmentation

This module provides three architectures:
1. U-Net (Baseline)
2. Attention U-Net
3. HybridUNet (SOTA: UNet3+ + ASPP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================================
# UNET (Baseline)
# ============================================================================

class Conv3x3(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3x3(channels_in, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            Conv3x3(channels_out, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DoubleConv(channels_in, channels_out)
        )
    
    def forward(self, x):
        return self.encoder(x)


class UpConv(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.Conv2d(channels_in, channels_in//2, kernel_size=1, stride=1)
        )
        self.decoder = DoubleConv(channels_in, channels_out)
    
    def forward(self, x1, x2):
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.decoder(x)


class UNET(nn.Module):
    def __init__(self, channels_in=1, channels=64, num_classes=1):
        super().__init__()
        self.first_conv = DoubleConv(channels_in, channels)
        self.down_conv1 = DownConv(channels, 2*channels)
        self.down_conv2 = DownConv(2*channels, 4*channels)
        self.down_conv3 = DownConv(4*channels, 8*channels)
        self.middle_conv = DownConv(8*channels, 16*channels)
        self.up_conv1 = UpConv(16*channels, 8*channels)
        self.up_conv2 = UpConv(8*channels, 4*channels)
        self.up_conv3 = UpConv(4*channels, 2*channels)
        self.up_conv4 = UpConv(2*channels, channels)
        self.last_conv = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)
        
    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        x5 = self.middle_conv(x4)
        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)
        return self.last_conv(u4)


# ============================================================================
# ATTENTION UNET
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=True, stride=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=True, stride=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=True, stride=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return psi * x


class AttentionUNet(nn.Module):
    def __init__(self, channels_in=1, channels_out=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(channels_in=channels_in, channels_out=64)
        self.conv2 = ConvBlock(channels_in=64, channels_out=128)
        self.conv3 = ConvBlock(channels_in=128, channels_out=256)
        self.conv4 = ConvBlock(channels_in=256, channels_out=512)
        self.conv5 = ConvBlock(channels_in=512, channels_out=1024)
        self.up5 = UpConvBlock(channels_in=1024, channels_out=512)
        self.att5 = AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.up_conv5 = ConvBlock(channels_in=1024, channels_out=512)
        self.up4 = UpConvBlock(channels_in=512, channels_out=256)
        self.att4 = AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.up_conv4 = ConvBlock(channels_in=512, channels_out=256)
        self.up3 = UpConvBlock(channels_in=256, channels_out=128)
        self.att3 = AttentionBlock(f_g=128, f_l=128, f_int=64)
        self.up_conv3 = ConvBlock(channels_in=256, channels_out=128)
        self.up2 = UpConvBlock(channels_in=128, channels_out=64)
        self.att2 = AttentionBlock(f_g=64, f_l=64, f_int=32)
        self.up_conv2 = ConvBlock(channels_in=128, channels_out=64)
        self.final_conv = nn.Conv2d(64, channels_out, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_conv5(d5)
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)
        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)
        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        d1 = self.final_conv(d2)
        return d1


# ============================================================================
# HYBRID UNET (SOTA: UNet3+ + ASPP)
# ============================================================================

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.conv_atrous_1 = self.create_atrous_branch(in_channels, out_channels // 4, rate=rates[1])
        self.conv_atrous_2 = self.create_atrous_branch(in_channels, out_channels // 4, rate=rates[2])
        self.conv_atrous_3 = self.create_atrous_branch(in_channels, out_channels // 4, rate=rates[3])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.Identity(),
            nn.ReLU(inplace=True)
        )
        total_out_channels = (out_channels // 4) * 5
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def create_atrous_branch(self, in_channels, out_channels, rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        b1 = self.conv_1x1(x)
        b2 = self.conv_atrous_1(x)
        b3 = self.conv_atrous_2(x)
        b4 = self.conv_atrous_3(x)
        b5 = self.gap(x)
        b5 = F.interpolate(b5, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.final_conv(x)


class HybridUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, use_aspp=True, dropout=0.0):
        super().__init__()
        self.use_aspp = use_aspp
        filters = [64, 128, 256, 512, 1024]
        cat_channels = 64
        decoder_in_channels = cat_channels * 5
        enc1_in_channels = in_channels
        self.enc1 = ConvBlock(enc1_in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = ConvBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2, 2)
        if self.use_aspp:
            self.bottleneck = ASPP(filters[3], filters[4])
        else:
            self.bottleneck = ConvBlock(filters[3], filters[4])
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.e1_to_d4 = self.create_scaler(filters[0], cat_channels, scale_factor=1/8)
        self.e2_to_d4 = self.create_scaler(filters[1], cat_channels, scale_factor=1/4)
        self.e3_to_d4 = self.create_scaler(filters[2], cat_channels, scale_factor=1/2)
        self.e4_to_d4 = self.create_scaler(filters[3], cat_channels, scale_factor=1)
        self.b_to_d4  = self.create_scaler(filters[4], cat_channels, scale_factor=2)
        self.dec4 = ConvBlock(decoder_in_channels, filters[3])
        self.e1_to_d3 = self.create_scaler(filters[0], cat_channels, scale_factor=1/4)
        self.e2_to_d3 = self.create_scaler(filters[1], cat_channels, scale_factor=1/2)
        self.e3_to_d3 = self.create_scaler(filters[2], cat_channels, scale_factor=1)
        self.d4_to_d3 = self.create_scaler(filters[3], cat_channels, scale_factor=2)
        self.b_to_d3  = self.create_scaler(filters[4], cat_channels, scale_factor=4)
        self.dec3 = ConvBlock(decoder_in_channels, filters[2])
        self.e1_to_d2 = self.create_scaler(filters[0], cat_channels, scale_factor=1/2)
        self.e2_to_d2 = self.create_scaler(filters[1], cat_channels, scale_factor=1)
        self.d3_to_d2 = self.create_scaler(filters[2], cat_channels, scale_factor=2)
        self.d4_to_d2 = self.create_scaler(filters[3], cat_channels, scale_factor=4)
        self.b_to_d2  = self.create_scaler(filters[4], cat_channels, scale_factor=8)
        self.dec2 = ConvBlock(decoder_in_channels, filters[1])
        self.e1_to_d1 = self.create_scaler(filters[0], cat_channels, scale_factor=1)
        self.d2_to_d1 = self.create_scaler(filters[1], cat_channels, scale_factor=2)
        self.d3_to_d1 = self.create_scaler(filters[2], cat_channels, scale_factor=4)
        self.d4_to_d1 = self.create_scaler(filters[3], cat_channels, scale_factor=8)
        self.b_to_d1  = self.create_scaler(filters[4], cat_channels, scale_factor=16)
        self.dec1 = ConvBlock(decoder_in_channels, filters[0])
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def create_scaler(self, in_channels, out_channels, scale_factor):
        layers = []
        if scale_factor < 1:
            pool_stride = int(1 / scale_factor)
            layers.append(nn.MaxPool2d(kernel_size=pool_stride, stride=pool_stride))
        elif scale_factor > 1:
            layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        b = self.dropout(b)
        d4_in = torch.cat([self.e1_to_d4(e1), self.e2_to_d4(e2), self.e3_to_d4(e3), self.e4_to_d4(e4), self.b_to_d4(b)], dim=1)
        d4 = self.dec4(d4_in)
        d3_in = torch.cat([self.e1_to_d3(e1), self.e2_to_d3(e2), self.e3_to_d3(e3), self.d4_to_d3(d4), self.b_to_d3(b)], dim=1)
        d3 = self.dec3(d3_in)
        d2_in = torch.cat([self.e1_to_d2(e1), self.e2_to_d2(e2), self.d3_to_d2(d3), self.d4_to_d2(d4), self.b_to_d2(b)], dim=1)
        d2 = self.dec2(d2_in)
        d1_in = torch.cat([self.e1_to_d1(e1), self.d2_to_d1(d2), self.d3_to_d1(d3), self.d4_to_d1(d4), self.b_to_d1(b)], dim=1)
        d1 = self.dec1(d1_in)
        return self.final_conv(d1)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_name: str, dropout: float = 0.0, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: One of 'unet', 'attention_unet', 'hybrid_unet'
        dropout: Dropout rate (only for hybrid_unet)
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    if model_name == 'unet':
        model = UNET(channels_in=1, channels=64, num_classes=1)
    elif model_name == 'attention_unet':
        model = AttentionUNet(channels_in=1, channels_out=1)
    elif model_name == 'hybrid_unet':
        # Allow overriding ASPP via kwargs for ablation studies
        use_aspp = kwargs.get('use_aspp', True)
        model = HybridUNet(in_channels=1, num_classes=1, use_aspp=use_aspp, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: unet, attention_unet, hybrid_unet")
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created {model_name} with {param_count:,} parameters")
    
    return model


if __name__ == "__main__":
    print("Model definitions loaded successfully")
    print("Available models: unet, attention_unet, hybrid_unet")
    print("Use create_model('model_name') to instantiate")

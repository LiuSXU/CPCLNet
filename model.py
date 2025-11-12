# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# 导入三大模块
from modules.CMDA import *
from modules.RAMP import *
from modules.CRPD import *

import config
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation


class CPCLNet(nn.Module):
    def __init__(self, num_fg_desc=102, num_bg_desc=102, feature_dim=1280):
        super(CPCLNet, self).__init__()

        # 1.特征提取器
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.feature_dim = feature_dim

        # 验证维度
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out_dim = self.encoder.extract_features(dummy).shape[1]
        assert out_dim == feature_dim, f"Expected {feature_dim}, got {out_dim}"

        # 2.三大模块
        self.crpd = CRPDModule(num_fg_desc, num_bg_desc, feature_dim)
        self.rwkv_affinity = RWKVAffinityModule(feature_dim=feature_dim, num_heads=8)
        self.aligner = DynamicAlignmentModule(feature_dim=feature_dim, proj_dim=256)

        # 3.边界原型生成
        self.register_buffer('kernel', torch.ones(3, 3))

        # 4.解码器（前景 + 背景）
        max_fg = num_fg_desc + 2  # + proto + boundary
        max_bg = num_bg_desc + 2

        def make_decoder(in_channels):
            return nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, 256, 3, padding=1),
                    nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout(0.3)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                ),
                nn.Conv2d(64, 1, 1)
            ])

        self.decoder_fg = make_decoder(max_fg + 2)
        self.decoder_bg = make_decoder(max_bg + 2)

        # 注意力机制
        self.attn_fg = nn.Sequential(nn.Conv2d(256, 1, 1), nn.Sigmoid())
        self.attn_bg = nn.Sequential(nn.Conv2d(256, 1, 1), nn.Sigmoid())

        # 最终融合
        self.fusion = nn.Conv2d(2, 2, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_boundary(self, mask):
        mask_np = mask.cpu().numpy()
        B = mask_np.shape[0]
        inner, outer = [], []
        for i in range(B):
            m = mask_np[i, 0]
            e = binary_erosion(m, structure=np.ones((3, 3)))
            d = binary_dilation(m, structure=np.ones((3, 3)))
            inner.append(m - e)
            outer.append(d - m)
        inner = torch.from_numpy(np.stack(inner)).float().to(mask.device).unsqueeze(1)
        outer = torch.from_numpy(np.stack(outer)).float().to(mask.device).unsqueeze(1)
        return inner, outer

    def map_pooling(self, feat, mask):
        mask = F.interpolate(mask, size=feat.shape[2:], mode='bilinear', align_corners=True)
        masked = feat * mask
        return masked.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-8)

    def forward(self, support_img, support_mask, query_img):
        device = support_img.device
        B = support_img.shape[0]

        # 特征提取
        s_feat = self.encoder.extract_features(support_img)  # [B, D, H, W]
        q_feat = self.encoder.extract_features(query_img)

        # 动态对齐模块
        s_proj, q_proj, align_loss = self.aligner(s_feat, q_feat)
        # 使用对齐后的特征
        s_feat, q_feat = s_proj, q_proj

        # 边界原型
        inner_boundary, outer_boundary = self.get_boundary(support_mask)
        inner_boundary = F.interpolate(inner_boundary, size=q_feat.shape[2:], mode='bilinear', align_corners=True)
        outer_boundary = F.interpolate(outer_boundary, size=q_feat.shape[2:], mode='bilinear', align_corners=True)

        ib_proto = self.map_pooling(s_feat, inner_boundary)
        ob_proto = self.map_pooling(s_feat, outer_boundary)

        # 聚类原型生成
        fg_desc, bg_desc = self.crpd(s_feat, support_mask)  # [K, D]

        # 全局平均原型
        fg_proto = self.map_pooling(s_feat, support_mask)
        bg_proto = self.map_pooling(s_feat, 1 - support_mask)

        # 拼接：聚类原型 + 全局原型 + 边界原型
        fg_desc = torch.cat([fg_desc, fg_proto, ib_proto], dim=0)
        bg_desc = torch.cat([bg_desc, bg_proto, ob_proto], dim=0)


        fg_affinity = self.rwkv_affinity(fg_desc, q_feat)  # [K_fg, D]
        bg_affinity = self.rwkv_affinity(bg_desc, q_feat)

        # 亲和力图生成
        H, W = q_feat.shape[2:]
        q_flat = q_feat.flatten(2)  # [B, D, HW]

        fg_map = torch.matmul(fg_affinity, q_flat).view(-1, B, H, W)  # [K_fg, B, H, W]
        bg_map = torch.matmul(bg_affinity, q_flat).view(-1, B, H, W)

        fg_map = fg_map.permute(1, 0, 2, 3)  # [B, K_fg, H, W]
        bg_map = bg_map.permute(1, 0, 2, 3)

        # 拼接边界
        fg_input = torch.cat([fg_map, inner_boundary, outer_boundary], dim=1)
        bg_input = torch.cat([bg_map, inner_boundary, outer_boundary], dim=1)


        x = self.decoder_fg[0](fg_input)
        attn = self.attn_fg(x)
        x = x * attn + x
        for layer in self.decoder_fg[1:]:
            x = layer(x)
        fg_pred = x

        x = self.decoder_bg[0](bg_input)
        attn = self.attn_bg(x)
        x = x * attn + x
        for layer in self.decoder_bg[1:]:
            x = layer(x)
        bg_pred = x

        # 融合
        pred = torch.cat([bg_pred, fg_pred], dim=1)
        pred = self.fusion(pred)

        return pred, fg_desc, bg_desc, q_feat, s_feat, align_loss
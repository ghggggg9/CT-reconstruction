# gan_model_transformer.py
"""
完整的Transformer GAN模型 - 修复版
包含生成器、判别器和所有必要的组件
修复了通道不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ============================================
# 基础模块和激活函数
# ============================================

class Swish(nn.Module):
    """Swish激活函数 - 比LeakyReLU更平滑"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GELU(nn.Module):
    """GELU激活函数 - Transformer常用"""
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ============================================
# 注意力模块
# ============================================

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """交叉注意力模块 - 用于2D到3D的特征融合"""
    def __init__(self, dim_q, dim_kv, num_heads=8, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.k = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.v = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, q, kv):
        B, N_q, C_q = q.shape
        B, N_kv, C_kv = kv.shape
        
        q = self.q(q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(kv).reshape(B, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================
# Transformer块
# ============================================

class TransformerBlock(nn.Module):
    """标准Transformer块"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, 
                 dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            dropout=attn_dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """带交叉注意力的Transformer块"""
    def __init__(self, dim_q, dim_kv, num_heads=8, mlp_ratio=4.0, 
                 qkv_bias=False, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.cross_attn = CrossAttention(
            dim_q, dim_kv, num_heads=num_heads, 
            qkv_bias=qkv_bias, dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim_q)
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim_q, mlp_hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim_q),
            nn.Dropout(dropout)
        )
    
    def forward(self, q, kv):
        q = q + self.cross_attn(self.norm_q(q), self.norm_kv(kv))
        q = q + self.mlp(self.norm2(q))
        return q


# ============================================
# 卷积模块
# ============================================

class ConvBlock3D(nn.Module):
    """增强的3D卷积块"""
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, 
                 use_norm=True, activation='swish', dropout=0.0):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size, stride, padding, bias=not use_norm)
        
        self.norm = None
        if use_norm:
            self.norm = nn.GroupNorm(min(32, out_c//2), out_c)
        
        self.activation = None
        if activation == 'swish':
            self.activation = Swish()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = GELU()
        
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResidualBlock3D(nn.Module):
    """3D残差块"""
    def __init__(self, in_c, out_c, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBlock3D(in_c, out_c, stride=stride, dropout=dropout)
        self.conv2 = ConvBlock3D(out_c, out_c, activation=None, dropout=dropout)
        
        self.shortcut = None
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Conv3d(in_c, out_c, 1, stride, bias=False)
        
        self.activation = Swish()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out = out + identity
        out = self.activation(out)
        return out


# ============================================
# 位置编码
# ============================================

class PositionalEncoding3D(nn.Module):
    """3D位置编码"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        
        # 创建位置编码
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0)
        z_embed = torch.arange(D, dtype=torch.float32, device=x.device).unsqueeze(1).unsqueeze(2)
        
        # 归一化到[-1, 1]
        y_embed = 2 * y_embed / (H - 1) - 1 if H > 1 else torch.zeros_like(y_embed)
        x_embed = 2 * x_embed / (W - 1) - 1 if W > 1 else torch.zeros_like(x_embed)
        z_embed = 2 * z_embed / (D - 1) - 1 if D > 1 else torch.zeros_like(z_embed)
        
        # 扩展维度
        y_embed = y_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, D, H, W)
        x_embed = x_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, 1, D, H, W)
        z_embed = z_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, D, H, W)
        
        # 生成sin/cos编码
        dim_t = torch.arange(C // 6, dtype=torch.float32, device=x.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (C // 6))
        
        pos_x = x_embed / dim_t.view(1, -1, 1, 1, 1)
        pos_y = y_embed / dim_t.view(1, -1, 1, 1, 1)
        pos_z = z_embed / dim_t.view(1, -1, 1, 1, 1)
        
        # 应用sin/cos
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1, 2)
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1, 2)
        pos_z = torch.stack([pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()], dim=2).flatten(1, 2)
        
        # 组合位置编码
        pos = torch.cat([pos_x[:, :C//3], pos_y[:, :C//3], pos_z[:, :C//3]], dim=1)
        
        # 确保通道数匹配
        if pos.shape[1] < C:
            padding = torch.zeros(B, C - pos.shape[1], D, H, W, device=x.device)
            pos = torch.cat([pos, padding], dim=1)
        else:
            pos = pos[:, :C]
        
        return x + pos


# ============================================
# Hybrid Transformer Generator
# ============================================

class HybridTransformerGenerator(nn.Module):
    """混合Transformer生成器 - 结合CNN和Transformer的优势"""
    def __init__(self, base_channels=32, max_depth=64, num_heads=8, 
                 num_transformer_blocks=6, dropout=0.1):
        super().__init__()
        self.max_depth = max_depth
        self.base_channels = base_channels
        
        # ========== 2D CNN编码器 ==========
        self.encoder_2d = nn.ModuleList([
            ConvBlock3D(2, base_channels, kernel_size=7, padding=3),  # 使用3D卷积处理2D+通道
            ResidualBlock3D(base_channels, base_channels*2, stride=2),
            ResidualBlock3D(base_channels*2, base_channels*4, stride=2),
            ResidualBlock3D(base_channels*4, base_channels*8, stride=2),
        ])
        
        # ========== 2D到3D投影 ==========
        self.projection = nn.Sequential(
            nn.Linear(base_channels*8, base_channels*8),
            nn.LayerNorm(base_channels*8),
            GELU(),
            nn.Linear(base_channels*8, max_depth * base_channels*4),
        )
        
        # ========== 位置编码 ==========
        self.pos_encoding = PositionalEncoding3D(base_channels*4)
        
        # ========== Transformer编码器 ==========
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=base_channels*4,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout
            ) for _ in range(num_transformer_blocks)
        ])
        
        # ========== 交叉注意力（2D特征到3D） ==========
        self.cross_attention = CrossAttentionBlock(
            dim_q=base_channels*4,
            dim_kv=base_channels*8,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ========== 3D CNN解码器 ==========
        self.decoder_3d = nn.ModuleList([
            ResidualBlock3D(base_channels*4, base_channels*4),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(base_channels*4, base_channels*2),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(base_channels*2, base_channels),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
            ResidualBlock3D(base_channels, base_channels),
        ])
        
        # ========== 输出层 ==========
        self.output = nn.Sequential(
            ConvBlock3D(base_channels, base_channels//2, kernel_size=3, padding=1),
            nn.Conv3d(base_channels//2, 1, kernel_size=1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, xray_pair, target_depth=None):
        """
        前向传播
        xray_pair: [B, 2, H, W] - AP和LAT视图
        target_depth: 目标深度
        """
        B, _, H, W = xray_pair.shape
        target_depth = target_depth or self.max_depth
        
        # ========== 2D特征提取 ==========
        # 将2D输入扩展为3D（添加单位深度维度）
        x = xray_pair.unsqueeze(2)  # [B, 2, 1, H, W]
        
        # 通过2D编码器
        features_2d = []
        for layer in self.encoder_2d:
            x = layer(x)
            features_2d.append(x)
        
        # 获取最后的2D特征
        feat_2d = x.squeeze(2)  # [B, C, H', W']
        B, C, H_feat, W_feat = feat_2d.shape
        
        # ========== 2D到3D投影 ==========
        # 全局池化获取全局特征
        global_feat = F.adaptive_avg_pool2d(feat_2d, 1).view(B, -1)  # [B, C]
        
        # 投影到3D空间
        depth_features = self.projection(global_feat)  # [B, max_depth*C']
        depth_features = depth_features.view(B, self.max_depth, self.base_channels*4)  # 使用max_depth

        # 如果需要不同的深度，裁剪或填充
        if target_depth != self.max_depth:
            if target_depth < self.max_depth:
                depth_features = depth_features[:, :target_depth, :]
            else:
                # 填充（通常不会发生）
                pad_size = target_depth - self.max_depth
                padding = depth_features[:, -1:, :].expand(B, pad_size, -1)
                depth_features = torch.cat([depth_features, padding], dim=1)
        
        # 创建3D特征图
        feat_3d = depth_features.unsqueeze(3).unsqueeze(4)  # [B, D, C', 1, 1]
        feat_3d = feat_3d.permute(0, 2, 1, 3, 4)  # [B, C', D, 1, 1]
        feat_3d = F.interpolate(feat_3d, size=(target_depth//8, H_feat, W_feat), 
                               mode='trilinear', align_corners=False)  # [B, C', D', H', W']
        
        # 添加位置编码
        feat_3d = self.pos_encoding(feat_3d)
        
        # ========== Transformer处理 ==========
        # 将3D特征转换为序列
        B, C_3d, D_3d, H_3d, W_3d = feat_3d.shape
        feat_seq = feat_3d.flatten(2).transpose(1, 2)  # [B, D*H*W, C]
        
        # 通过Transformer块
        for block in self.transformer_blocks:
            feat_seq = block(feat_seq)
        
        # ========== 交叉注意力 ==========
        # 准备2D特征序列
        feat_2d_seq = feat_2d.flatten(2).transpose(1, 2)  # [B, H'*W', C_2d]
        
        # 应用交叉注意力
        feat_seq = self.cross_attention(feat_seq, feat_2d_seq)
        
        # 重塑回3D
        feat_3d = feat_seq.transpose(1, 2).reshape(B, C_3d, D_3d, H_3d, W_3d)
        
        # ========== 3D CNN解码器 ==========
        for layer in self.decoder_3d:
            feat_3d = layer(feat_3d)
        
        # 调整到目标尺寸
        if feat_3d.shape[2:] != (target_depth, H, W):
            feat_3d = F.interpolate(feat_3d, size=(target_depth, H, W), 
                                  mode='trilinear', align_corners=False)
        
        # ========== 输出 ==========
        output = self.output(feat_3d)
        
        return output


# ============================================
# 判别器 - 修复版本
# ============================================

# 替换 gan_model_transformer.py 中的 PatchDiscriminator3D 类

class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN判别器 - 自适应版本，解决所有通道不匹配问题"""
    def __init__(self, base_channels=64, num_layers=4):
        super().__init__()
        
        # 保存配置
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # X光编码器 - 明确定义输出通道数
        self.xray_output_channels = base_channels * 2
        
        # X光编码器网络
        self.xray_encoder = nn.Sequential(
            nn.Conv2d(2, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, max(1, base_channels)), base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels*2, self.xray_output_channels, 3, stride=1, padding=1),
            nn.GroupNorm(min(32, max(1, base_channels)), self.xray_output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 预期的输入通道数：1个CT通道 + xray_output_channels
        self.expected_input_channels = 1 + self.xray_output_channels
        
        # 构建判别器主体（使用ModuleList以支持动态第一层）
        self.discriminator_layers = nn.ModuleList()
        
        # 第一层会动态创建以匹配实际输入通道数
        self.first_layer = None
        
        # 构建其余层
        in_channels = base_channels
        out_channels = base_channels
        
        for i in range(1, num_layers):
            stride = 2 if i < num_layers - 1 else 1
            out_channels = min(in_channels * 2, 512)
            
            block = []
            block.append(nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1))
            
            if i > 1:  # 第二层之后添加normalization
                num_groups = min(32, max(1, out_channels//2))
                block.append(nn.GroupNorm(num_groups, out_channels))
            
            block.append(nn.LeakyReLU(0.2, inplace=True))
            
            self.discriminator_layers.append(nn.Sequential(*block))
            in_channels = out_channels
        
        # 输出层
        self.output_layer = nn.Conv3d(in_channels, 1, 3, padding=1)
        
        print(f"[Discriminator] 初始化完成:")
        print(f"  - base_channels: {base_channels}")
        print(f"  - xray编码器输出: {self.xray_output_channels}通道")
        print(f"  - 期望总输入: {self.expected_input_channels}通道")
    
    def _create_first_layer(self, actual_channels):
        """动态创建第一层以匹配实际输入通道数"""
        if self.first_layer is None or self.first_layer.in_channels != actual_channels:
            device = next(self.parameters()).device
            self.first_layer = nn.Conv3d(
                actual_channels, self.base_channels, 
                3, stride=2, padding=1
            ).to(device)
            
            # 初始化权重
            nn.init.kaiming_normal_(self.first_layer.weight, mode='fan_out', 
                                   nonlinearity='leaky_relu', a=0.2)
            if self.first_layer.bias is not None:
                nn.init.constant_(self.first_layer.bias, 0)
            
            print(f"[Discriminator] 创建新的第一层: {actual_channels} -> {self.base_channels}")
    
    def forward(self, xray_pair, ct_volume):
        """
        xray_pair: [B, 2, H, W]
        ct_volume: [B, 1, D, H, W]
        """
        B, C_ct, D, H, W = ct_volume.shape
        
        # 验证CT通道数
        if C_ct != 1:
            print(f"[WARNING] CT有{C_ct}个通道而不是1个!")
        
        # 编码X光特征
        xray_feat = self.xray_encoder(xray_pair)  # [B, xray_output_channels, H', W']
        actual_xray_channels = xray_feat.shape[1]
        
        # 验证X光编码器输出
        if actual_xray_channels != self.xray_output_channels:
            print(f"[WARNING] X光编码器输出{actual_xray_channels}通道，期望{self.xray_output_channels}通道")
        
        # 计算目标尺寸
        target_h = xray_feat.shape[2]
        target_w = xray_feat.shape[3]
        target_d = max(D // 4, 8)
        
        # 扩展X光特征到3D
        xray_feat_3d = xray_feat.unsqueeze(2).expand(-1, -1, target_d, -1, -1)
        
        # 下采样CT体积
        ct_down = F.interpolate(ct_volume, size=(target_d, target_h, target_w), 
                               mode='trilinear', align_corners=False)
        
        # 拼接
        combined = torch.cat([ct_down, xray_feat_3d], dim=1)
        actual_channels = combined.shape[1]
        
        # 如果通道数不匹配，打印详细信息并尝试修复
        if actual_channels != self.expected_input_channels:
            print(f"\n[Discriminator] 通道数不匹配:")
            print(f"  CT通道: {ct_down.shape[1]}")
            print(f"  X光通道: {xray_feat_3d.shape[1]}")
            print(f"  实际总通道: {actual_channels}")
            print(f"  期望总通道: {self.expected_input_channels}")
            print(f"  差异: {actual_channels - self.expected_input_channels}")
            
            # 尝试自动修复
            if abs(actual_channels - self.expected_input_channels) <= 2:
                if actual_channels > self.expected_input_channels:
                    # 移除多余的通道
                    combined = combined[:, :self.expected_input_channels]
                    print(f"  [FIX] 移除{actual_channels - self.expected_input_channels}个多余通道")
                else:
                    # 添加零填充通道
                    padding_channels = self.expected_input_channels - actual_channels
                    padding = torch.zeros(B, padding_channels, target_d, target_h, target_w, 
                                         device=combined.device, dtype=combined.dtype)
                    combined = torch.cat([combined, padding], dim=1)
                    print(f"  [FIX] 添加{padding_channels}个零填充通道")
                actual_channels = combined.shape[1]
        
        # 动态创建或调整第一层
        self._create_first_layer(actual_channels)
        
        # 前向传播
        x = self.first_layer(combined)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # 通过其余层
        for layer in self.discriminator_layers:
            x = layer(x)
        
        # 输出
        output = self.output_layer(x)
        
        return output


# ============================================
# 损失函数
# ============================================

class ProjectionConsistencyLoss(nn.Module):
    """投影一致性损失 - 确保生成的CT投影与输入X光一致"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        self.l1_loss = nn.L1Loss()
    
    def forward(self, generated_ct, input_xrays):
        """
        generated_ct: [B, 1, D, H, W]
        input_xrays: [B, 2, H, W] - AP和LAT视图
        """
        # AP投影（前后位） - 沿深度维度求平均
        ap_proj = generated_ct.mean(dim=2)  # [B, 1, H, W]
        
        # LAT投影（侧位） - 沿宽度维度求平均
        lat_proj = generated_ct.mean(dim=4)  # [B, 1, D, H]
        lat_proj = lat_proj.permute(0, 1, 3, 2)  # [B, 1, H, D]
        
        # 调整投影尺寸以匹配输入
        if lat_proj.shape[-1] != input_xrays.shape[-1]:
            lat_proj = F.interpolate(lat_proj, size=input_xrays.shape[-2:], 
                                   mode='bilinear', align_corners=False)
        
        # 归一化投影到[-1, 1]
        ap_proj = self._normalize(ap_proj)
        lat_proj = self._normalize(lat_proj)
        
        # 计算损失
        ap_loss = self.l1_loss(ap_proj, input_xrays[:, 0:1])
        lat_loss = self.l1_loss(lat_proj, input_xrays[:, 1:2])
        
        return self.weight * (ap_loss + lat_loss)
    
    def _normalize(self, x):
        """归一化到[-1, 1]"""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        x_min = x_flat.min(dim=-1, keepdim=True)[0]
        x_max = x_flat.max(dim=-1, keepdim=True)[0]
        x_normalized = 2 * (x_flat - x_min) / (x_max - x_min + 1e-8) - 1
        return x_normalized.view(B, C, H, W)


class PerceptualLoss(nn.Module):
    """感知损失 - 使用预训练的2D网络计算"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        # 这里可以使用预训练的VGG等网络
        # 简化版本使用多尺度L1
        self.l1_loss = nn.L1Loss()
    
    def forward(self, generated_ct, real_ct):
        """计算多尺度感知损失"""
        loss = 0
        
        # 在不同尺度计算损失
        for scale in [1, 2, 4]:
            if scale > 1:
                generated_scaled = F.avg_pool3d(generated_ct, scale)
                real_scaled = F.avg_pool3d(real_ct, scale)
            else:
                generated_scaled = generated_ct
                real_scaled = real_ct
            
            loss += self.l1_loss(generated_scaled, real_scaled)
        
        return self.weight * loss / 3.0


# ============================================
# 辅助函数
# ============================================

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_models():
    """测试模型是否正常工作"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # 测试输入
    batch_size = 1
    xray = torch.randn(batch_size, 2, 256, 256).to(device)
    ct_real = torch.randn(batch_size, 1, 48, 256, 256).to(device)
    
    # 测试生成器
    print("\n" + "="*60)
    print("Testing Hybrid Transformer Generator...")
    print("="*60)
    G = HybridTransformerGenerator(base_channels=24, max_depth=48).to(device)
    with torch.no_grad():
        ct_fake = G(xray, target_depth=48)
    print(f"✓ Generator output shape: {ct_fake.shape}")
    print(f"  Parameters: {count_parameters(G):,}")
    
    # 测试判别器
    print("\n" + "="*60)
    print("Testing Patch Discriminator...")
    print("="*60)
    D = PatchDiscriminator3D(base_channels=48).to(device)
    with torch.no_grad():
        score_real = D(xray, ct_real)
        score_fake = D(xray, ct_fake)
    print(f"✓ Discriminator output shape: {score_real.shape}")
    print(f"  Parameters: {count_parameters(D):,}")
    
    # 测试损失函数
    print("\n" + "="*60)
    print("Testing Loss Functions...")
    print("="*60)
    
    # 投影一致性损失
    proj_loss = ProjectionConsistencyLoss()
    loss_proj = proj_loss(ct_fake, xray)
    print(f"✓ Projection consistency loss: {loss_proj.item():.4f}")
    
    # 感知损失
    percep_loss = PerceptualLoss()
    loss_percep = percep_loss(ct_fake, ct_real)
    print(f"✓ Perceptual loss: {loss_percep.item():.4f}")
    
    print("\n" + "="*60)
    print("All models tested successfully!")
    print("="*60)


if __name__ == "__main__":
    test_models()
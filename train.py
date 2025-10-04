#!/usr/bin/env python3
"""
Complete Fixed Transformer GAN Training Script with Checkpoint Resume
处理通道不匹配问题的完整版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from PIL import Image
import os
import sys
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
from datetime import datetime
import warnings
import random
import math
import gc
warnings.filterwarnings('ignore')

# Safe memory optimization that works with all PyTorch versions
try:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
except:
    pass

# Import models
from gan_model_transformer import (
    HybridTransformerGenerator,
    PatchDiscriminator3D,
    ProjectionConsistencyLoss,
    PerceptualLoss,
    count_parameters
)


# ============================================
# Dataset Class
# ============================================

class CTDataset(Dataset):
    """CT Reconstruction Dataset - Optimized Version"""
    def __init__(self, drr_dir, ct_dir, target_size=256, max_depth=48, 
                 augmentation=True, training=True):
        self.drr_dir = drr_dir
        self.ct_dir = ct_dir
        self.target_size = target_size
        self.max_depth = max_depth
        self.augmentation = augmentation and training
        self.training = training
        
        # Get valid patients
        self.patients = []
        if os.path.exists(drr_dir) and os.path.exists(ct_dir):
            for pid in os.listdir(drr_dir):
                drr_path = os.path.join(drr_dir, pid)
                ct_path = os.path.join(ct_dir, pid)
                
                # Support multiple DRR file naming conventions
                ap_paths = [
                    os.path.join(drr_path, 'drr_ap.png'),
                    os.path.join(drr_path, 'ap.png'),
                    os.path.join(drr_path, 'AP.png'),
                    os.path.join(drr_path, 'coronal_view.png'),
                ]
                lat_paths = [
                    os.path.join(drr_path, 'drr_lat.png'),
                    os.path.join(drr_path, 'lat.png'),
                    os.path.join(drr_path, 'LAT.png'),
                    os.path.join(drr_path, 'sagittal_view.png'),
                ]
                
                ap_exists = any(os.path.exists(p) for p in ap_paths)
                lat_exists = any(os.path.exists(p) for p in lat_paths)
                
                if ap_exists and lat_exists and os.path.exists(ct_path):
                    # Check if there are enough DICOM files
                    dcm_files = [f for f in os.listdir(ct_path) if f.endswith('.dcm')]
                    if len(dcm_files) >= 16:  # Need at least 16 slices
                        self.patients.append(pid)
        
        print(f"Found {len(self.patients)} valid patients")
        if len(self.patients) == 0:
            raise ValueError("No valid patients found! Check data paths.")
    
    def __len__(self):
        return len(self.patients)
    
    # 替换train.py中的CTDataset.__getitem__方法

    def __getitem__(self, idx):
        pid = self.patients[idx % len(self.patients)]
        
        try:
            # Load X-ray images
            drr_path = os.path.join(self.drr_dir, pid)
            
            # Try different file names
            ap_paths = [
                os.path.join(drr_path, 'drr_ap.png'),
                os.path.join(drr_path, 'ap.png'),
                os.path.join(drr_path, 'AP.png'),
                os.path.join(drr_path, 'coronal_view.png'),
            ]
            lat_paths = [
                os.path.join(drr_path, 'drr_lat.png'),
                os.path.join(drr_path, 'lat.png'),
                os.path.join(drr_path, 'LAT.png'),
                os.path.join(drr_path, 'sagittal_view.png'),
            ]
            
            ap_path = next((p for p in ap_paths if os.path.exists(p)), None)
            lat_path = next((p for p in lat_paths if os.path.exists(p)), None)
            
            if not ap_path or not lat_path:
                raise FileNotFoundError(f"X-ray files not found for {pid}")
            
            # Load and preprocess X-rays
            ap = Image.open(ap_path).convert('L')
            lat = Image.open(lat_path).convert('L')
            
            ap = ap.resize((self.target_size, self.target_size), Image.LANCZOS)
            lat = lat.resize((self.target_size, self.target_size), Image.LANCZOS)
            
            ap = np.array(ap, dtype=np.float32)
            lat = np.array(lat, dtype=np.float32)
            
            # Data augmentation
            if self.augmentation:
                # Random horizontal flip
                if random.random() > 0.5:
                    ap = np.fliplr(ap)
                    lat = np.fliplr(lat)
                    # Ensure contiguous memory
                    ap = np.ascontiguousarray(ap)
                    lat = np.ascontiguousarray(lat)
                
                # Random noise
                if random.random() > 0.7:
                    noise_level = random.uniform(0.01, 0.02)
                    ap = ap + np.random.randn(*ap.shape) * noise_level * 255
                    lat = lat + np.random.randn(*lat.shape) * noise_level * 255
                    ap = np.clip(ap, 0, 255)
                    lat = np.clip(lat, 0, 255)
            
            # Normalize to [-1, 1] and ensure contiguous memory
            ap = torch.from_numpy(np.ascontiguousarray(ap)).float() / 127.5 - 1
            lat = torch.from_numpy(np.ascontiguousarray(lat)).float() / 127.5 - 1
            xray = torch.stack([ap, lat])
            
            # Load CT slices
            ct_path = os.path.join(self.ct_dir, pid)
            slices = []
            slice_spacings = [3.0, 1.0, 1.0]
            
            dcm_files = sorted([f for f in os.listdir(ct_path) if f.endswith('.dcm')])
            dcm_files = dcm_files[:self.max_depth]  # Limit slice count
            
            for i, dcm_file in enumerate(dcm_files):
                try:
                    dcm_path = os.path.join(ct_path, dcm_file)
                    dcm = pydicom.dcmread(dcm_path)
                    img = dcm.pixel_array.astype(np.float32)
                    
                    # Get spacing info
                    if i == 0:
                        try:
                            pixel_spacing = dcm.PixelSpacing
                            slice_thickness = getattr(dcm, 'SliceThickness', 3.0)
                            slice_spacings = [
                                float(slice_thickness), 
                                float(pixel_spacing[0]), 
                                float(pixel_spacing[1])
                            ]
                        except:
                            pass
                    
                    # Window/level (使用更宽的窗口)
                    # 改为软组织窗口，更适合全身CT
                    img = np.clip(img, -400, 400)  # 软组织窗口
                    img = (img + 400) / 800  # 归一化到[0,1]
                    
                    # Resize
                    img_pil = Image.fromarray((img * 255).astype(np.uint8))
                    img_pil = img_pil.resize((self.target_size, self.target_size), Image.LANCZOS)
                    img_array = np.array(img_pil, dtype=np.float32) / 127.5 - 1
                    slices.append(img_array)
                except Exception as e:
                    continue
            
            if len(slices) < 16:
                return self.__getitem__((idx + 1) % len(self))
            
            # Stack slices - 确保是3D数组 [D, H, W]
            ct = np.stack(slices)
            
            # CT data augmentation
            if self.augmentation and random.random() > 0.5:
                ct = np.flip(ct, axis=2)
                ct = np.ascontiguousarray(ct)
            
            # 转换为tensor并添加通道维度
            ct = torch.from_numpy(np.ascontiguousarray(ct)).float()
            
            # 确保CT是 [1, D, H, W] 形状
            if ct.dim() == 3:  # 如果是 [D, H, W]
                ct = ct.unsqueeze(0)  # 变成 [1, D, H, W]
            
            # 验证通道数
            assert ct.shape[0] == 1, f"CT应该有1个通道，但有{ct.shape[0]}个通道"
            
            return xray, ct, pid, slice_spacings
            
        except Exception as e:
            print(f"Error loading {pid}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# 替换train.py中的collate_fn函数

def collate_fn(batch):
    """更健壮的批处理函数"""
    xrays, cts, pids, spacings = zip(*batch)
    
    target_depth = 48
    
    cts_fixed = []
    for i, ct in enumerate(cts):
        original_shape = ct.shape
        
        # 处理异常形状
        if ct.dim() == 3:  # [D, H, W]
            ct = ct.unsqueeze(0)  # -> [1, D, H, W]
        
        # 如果有多个通道（不应该发生，但以防万一）
        if ct.shape[0] > 1:
            print(f"[FIX] {pids[i]}: 检测到{ct.shape[0]}通道，合并为1")
            ct = ct.mean(dim=0, keepdim=True)
        
        # 处理深度
        current_depth = ct.shape[1]
        if current_depth != target_depth:
            if current_depth > target_depth:
                start = (current_depth - target_depth) // 2
                ct = ct[:, start:start+target_depth]
            else:
                pad = (target_depth - current_depth) // 2
                ct = F.pad(ct, (0,0,0,0,0,0,pad,target_depth-current_depth-pad))
        
        # 处理空间尺寸
        if ct.shape[2:] != (256, 256):
            print(f"[FIX] {pids[i]}: 调整空间尺寸从{ct.shape[2:]}到(256,256)")
            ct = F.interpolate(
                ct.unsqueeze(0), 
                size=(target_depth, 256, 256), 
                mode='trilinear', 
                align_corners=False
            ).squeeze(0)
        
        # 最终形状验证
        if ct.shape != (1, target_depth, 256, 256):
            print(f"[WARNING] {pids[i]}: 最终形状{ct.shape}不正确，强制调整")
            # 强制调整到正确形状
            ct_temp = torch.zeros(1, target_depth, 256, 256)
            # 复制尽可能多的数据
            min_c = min(ct.shape[0], 1)
            min_d = min(ct.shape[1], target_depth)
            min_h = min(ct.shape[2], 256)
            min_w = min(ct.shape[3], 256)
            ct_temp[:min_c, :min_d, :min_h, :min_w] = ct[:min_c, :min_d, :min_h, :min_w]
            ct = ct_temp
        
        cts_fixed.append(ct)
    
    return torch.stack(xrays), torch.stack(cts_fixed), pids, spacings[0]


# ============================================
# Improved Trainer Class
# ============================================

class ImprovedTransformerTrainer:
    """Improved Transformer Trainer - Prevents Mode Collapse"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        print("\n" + "="*60)
        print("Initializing Improved Transformer GAN Trainer")
        print("="*60)
        
        # Initialize models
        self.G = HybridTransformerGenerator(
            base_channels=config['base_channels'],
            max_depth=config['max_depth'],
            num_heads=config['num_heads'],
            num_transformer_blocks=config['num_transformer_blocks'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.D = PatchDiscriminator3D(
            base_channels=config['disc_channels'],
            num_layers=config['disc_layers']
        ).to(self.device)
        
        print(f"✓ Generator parameters: {count_parameters(self.G):,}")
        print(f"✓ Discriminator parameters: {count_parameters(self.D):,}")
        
        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_proj = ProjectionConsistencyLoss(weight=1.0)
        
        # Optimizers - note different learning rates
        self.opt_G = optim.AdamW(
            self.G.parameters(),
            lr=config['lr_g'],
            betas=(0.5, 0.999),
            weight_decay=config['weight_decay']
        )
        
        self.opt_D = optim.AdamW(
            self.D.parameters(),
            lr=config['lr_d'],
            betas=(0.5, 0.999),
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduling
        self.warmup_steps = config['warmup_steps']
        self.current_step = 0
        
        # Mixed precision
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # Training statistics
        self.step = 0
        self.start_epoch = 0
        self.d_losses = []
        self.g_losses = []
        
        # Best metrics tracking
        self.best_psnr = 0
        self.best_ssim = 0
        
        # Initialize weights
        self._init_weights()
        
        print(f"✓ Training strategy:")
        print(f"  - Discriminator update frequency: every {config['d_update_freq']} steps")
        print(f"  - L1 weight: {config['weight_l1']}")
        print(f"  - GAN weight: {config['weight_gan']}")
        print(f"  - Projection weight: {config['weight_proj']}")
    
    def _init_weights(self):
        """Better weight initialization"""
        def init_func(m):
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.G.apply(init_func)
        self.D.apply(init_func)
    
    def adjust_learning_rate(self):
        """Learning rate warmup and adjustment"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.current_step + 1) / self.warmup_steps
            for param_group in self.opt_G.param_groups:
                param_group['lr'] = self.config['lr_g'] * lr_scale
            for param_group in self.opt_D.param_groups:
                param_group['lr'] = self.config['lr_d'] * lr_scale
    
    def train_step(self, xray, ct_real):
        """Improved training step - prevents mode collapse"""
        self.step += 1
        self.current_step += 1
        
        # Adjust learning rate
        self.adjust_learning_rate()
        
        depth = ct_real.shape[2]
        
        # ========== Discriminator Training (Controlled Frequency) ==========
        loss_d = torch.tensor(0.0, device=self.device)
        
        # Only update discriminator at specific steps
        if self.step % self.config['d_update_freq'] == 0:
            self.opt_D.zero_grad()
            
            if self.config['use_amp']:
                with autocast():
                    # Real samples
                    pred_real = self.D(xray, ct_real)
                    # Label smoothing + noise
                    real_labels = torch.ones_like(pred_real) * 0.9 + torch.rand_like(pred_real) * 0.1
                    loss_d_real = self.criterion_gan(pred_real, real_labels)
                    
                    # Generate fake samples
                    with torch.no_grad():
                        ct_fake = self.G(xray, depth)
                    
                    pred_fake = self.D(xray, ct_fake.detach())
                    fake_labels = torch.zeros_like(pred_fake) + torch.rand_like(pred_fake) * 0.1
                    loss_d_fake = self.criterion_gan(pred_fake, fake_labels)
                    
                    loss_d = (loss_d_real + loss_d_fake) * 0.5
                
                self.scaler.scale(loss_d).backward()
                self.scaler.unscale_(self.opt_D)
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.config['grad_clip'])
                self.scaler.step(self.opt_D)
                self.scaler.update()
            else:
                # Non-AMP version
                pred_real = self.D(xray, ct_real)
                real_labels = torch.ones_like(pred_real) * 0.9
                loss_d_real = self.criterion_gan(pred_real, real_labels)
                
                with torch.no_grad():
                    ct_fake = self.G(xray, depth)
                
                pred_fake = self.D(xray, ct_fake.detach())
                fake_labels = torch.zeros_like(pred_fake) + 0.1
                loss_d_fake = self.criterion_gan(pred_fake, fake_labels)
                
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.config['grad_clip'])
                self.opt_D.step()
            
            self.d_losses.append(loss_d.item())
        
        # ========== Generator Training (Every Step) ==========
        self.opt_G.zero_grad()
        
        if self.config['use_amp']:
            with autocast():
                ct_fake = self.G(xray, depth)
                if self.step % 100 == 0:
                    print(f"生成器输出统计:")
                    print(f"  范围: [{ct_fake.min():.3f}, {ct_fake.max():.3f}]")
                    print(f"  均值: {ct_fake.mean():.3f}")
                    print(f"  标准差: {ct_fake.std():.3f}")
                    print(f"  饱和像素比例: {(ct_fake.abs() > 0.9).float().mean():.2%}")
                # GAN loss (very low weight!)
                pred_fake = self.D(xray, ct_fake)
                loss_g_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
                
                # L1 loss (main loss!)
                loss_g_l1 = self.criterion_l1(ct_fake, ct_real)
                
                # Projection consistency loss
                loss_g_proj = self.criterion_proj(ct_fake, xray)
                
                # Prevent all-white/all-black regularization
                content_mean = torch.abs(ct_fake.mean())
                content_std = ct_fake.std()
                content_reg = torch.abs(content_mean - ct_real.mean()) + torch.abs(content_std - ct_real.std())
                
                # Total loss
                loss_g = (
                    loss_g_gan * self.config['weight_gan'] +
                    loss_g_l1 * self.config['weight_l1'] +
                    loss_g_proj * self.config['weight_proj'] +
                    content_reg * 0.1
                )
            
            self.scaler.scale(loss_g).backward()
            self.scaler.unscale_(self.opt_G)
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config['grad_clip'])
            self.scaler.step(self.opt_G)
            self.scaler.update()
        else:
            ct_fake = self.G(xray, depth)
            if self.step % 100 == 0:
                print(f"生成器输出统计:")
                print(f"  范围: [{ct_fake.min():.3f}, {ct_fake.max():.3f}]")
                print(f"  均值: {ct_fake.mean():.3f}")
                print(f"  标准差: {ct_fake.std():.3f}")
                print(f"  饱和像素比例: {(ct_fake.abs() > 0.9).float().mean():.2%}")
            pred_fake = self.D(xray, ct_fake)
            loss_g_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = self.criterion_l1(ct_fake, ct_real)
            loss_g_proj = self.criterion_proj(ct_fake, xray)
            
            content_mean = torch.abs(ct_fake.mean())
            content_std = ct_fake.std()
            content_reg = torch.abs(content_mean - ct_real.mean()) + torch.abs(content_std - ct_real.std())
            
            loss_g = (
                loss_g_gan * self.config['weight_gan'] +
                loss_g_l1 * self.config['weight_l1'] +
                loss_g_proj * self.config['weight_proj'] +
                content_reg * 0.1
            )
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config['grad_clip'])
            self.opt_G.step()
        
        self.g_losses.append(loss_g.item())
        
        # Calculate metrics
        with torch.no_grad():
            metrics = compute_metrics(ct_real, ct_fake)
        
        return {
            'D': loss_d.item(),
            'G': loss_g.item(),
            'L1': loss_g_l1.item(),
            'Proj': loss_g_proj.item(),
            'PSNR': metrics['psnr'],
            'SSIM': metrics['ssim'],
            'lr_g': self.opt_G.param_groups[0]['lr'],
            'lr_d': self.opt_D.param_groups[0]['lr']
        }
    
    def save_checkpoint(self, epoch, path):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'config': self.config,
            'd_losses': self.d_losses[-1000:],  # Save recent losses
            'g_losses': self.g_losses[-1000:],
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
        }
        
        if self.scaler is not None:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved: {path}")


# ============================================
# 修复后的Checkpoint Loading Function
# ============================================

def load_checkpoint(trainer, checkpoint_path, device):
    """Load checkpoint for resume training with improved mismatch handling
    处理模型权重不匹配问题，特别是判别器的通道数不匹配
    """
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 尝试加载生成器权重
            generator_loaded = False
            try:
                trainer.G.load_state_dict(checkpoint['G'])
                print("  ✓ 成功加载生成器权重")
                generator_loaded = True
            except RuntimeError as e:
                print(f"  ⚠ 生成器权重不匹配: {e}")
                print("  → 保持新初始化的生成器权重")
            
            # 尝试加载判别器权重 - 如果失败，完全重新初始化
            discriminator_loaded = False
            try:
                # First, try to load the discriminator state
                trainer.D.load_state_dict(checkpoint['D'])
                print("  ✓ 成功加载判别器权重")
                discriminator_loaded = True
            except RuntimeError as e:
                print(f"  ⚠ 判别器权重不匹配: {e}")
                print("  → 完全重新初始化判别器")
                
                # Clear any partially loaded state
                del trainer.D
                torch.cuda.empty_cache()
                
                # Completely reinitialize the discriminator
                from gan_model_transformer import PatchDiscriminator3D
                trainer.D = PatchDiscriminator3D(
                    base_channels=trainer.config['disc_channels'],
                    num_layers=trainer.config['disc_layers']
                ).to(device)
                
                # Apply weight initialization
                def init_func(m):
                    if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.trunc_normal_(m.weight, std=0.02)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                
                trainer.D.apply(init_func)
                
                # Reinitialize discriminator optimizer
                trainer.opt_D = optim.AdamW(
                    trainer.D.parameters(),
                    lr=trainer.config['lr_d'],
                    betas=(0.5, 0.999),
                    weight_decay=trainer.config['weight_decay']
                )
                print("  ✓ 判别器完全重新初始化")
            
            # 加载优化器状态
            if generator_loaded:
                try:
                    trainer.opt_G.load_state_dict(checkpoint['opt_G'])
                    print("  ✓ 成功加载生成器优化器状态")
                except Exception as e:
                    print(f"  ⚠ 无法加载生成器优化器状态: {e}")
                    print("  → 使用新的优化器")
            
            # Only load discriminator optimizer if discriminator was successfully loaded
            if discriminator_loaded:
                try:
                    trainer.opt_D.load_state_dict(checkpoint['opt_D'])
                    print("  ✓ 成功加载判别器优化器状态")
                except Exception as e:
                    print(f"  ⚠ 无法加载判别器优化器状态: {e}")
                    print("  → 使用新的优化器")
            
            # 加载训练信息
            start_epoch = checkpoint.get('epoch', 0) + 1
            trainer.step = checkpoint.get('step', 0)
            trainer.current_step = trainer.step
            
            # 加载最佳指标
            trainer.best_psnr = checkpoint.get('best_psnr', 0)
            trainer.best_ssim = checkpoint.get('best_ssim', 0)
            
            # 加载AMP scaler（如果使用）
            if trainer.scaler and 'scaler' in checkpoint:
                try:
                    trainer.scaler.load_state_dict(checkpoint['scaler'])
                    print("  ✓ 成功加载AMP scaler")
                except Exception as e:
                    print(f"  ⚠ 无法加载AMP scaler: {e}")
                    print("  → 使用新的scaler")
                    trainer.scaler = GradScaler()
            
            # 加载损失历史（如果有）
            if 'd_losses' in checkpoint:
                trainer.d_losses = checkpoint.get('d_losses', [])
            if 'g_losses' in checkpoint:
                trainer.g_losses = checkpoint.get('g_losses', [])
            
            print(f"  ✓ 从epoch {checkpoint.get('epoch', 0)}恢复，步数: {trainer.step}")
            print(f"  ✓ 最佳指标 - PSNR: {trainer.best_psnr:.2f} dB, SSIM: {trainer.best_ssim:.3f}")
            
            # Clear GPU cache after loading
            torch.cuda.empty_cache()
            
            return start_epoch
            
        except Exception as e:
            print(f"  ✗ 加载checkpoint时出错: {e}")
            print("  改为从头开始训练")
            
            # Clean up and start fresh
            torch.cuda.empty_cache()
            return 0
    else:
        print(f"  ⚠ 未找到checkpoint: {checkpoint_path}")
        print("  从头开始训练")
        return 0


# ============================================
# Visualization Functions
# ============================================

def visualize_results(xray, ct_real, ct_fake, save_path, epoch, batch_idx, metrics=None):
    """改进的可视化 - 显示真实和生成的所有三个切面"""
    
    if ct_fake.shape != ct_real.shape:
        ct_fake = F.interpolate(ct_fake, size=ct_real.shape[2:], 
                               mode='trilinear', align_corners=False)
    
    # 转换到[0,1]范围
    xray = (xray[0].detach().cpu() + 1) / 2
    real = (ct_real[0, 0].detach().cpu() + 1) / 2
    fake = (ct_fake[0, 0].detach().cpu() + 1) / 2
    
    # 裁剪极值，防止全白或全黑
    real = torch.clamp(real, 0, 1)
    fake = torch.clamp(fake, 0, 1)
    
    D, H, W = real.shape
    
    # 创建更大的图形，4行6列
    fig = plt.figure(figsize=(24, 16))
    
    # ========== 第一行：输入X光和投影 ==========
    # AP X-ray输入
    ax1 = plt.subplot(4, 6, 1)
    ax1.imshow(xray[0], cmap='gray', vmin=0, vmax=1)
    ax1.set_title('AP X-ray Input', fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # LAT X-ray输入
    ax2 = plt.subplot(4, 6, 2)
    ax2.imshow(xray[1], cmap='gray', vmin=0, vmax=1)
    ax2.set_title('LAT X-ray Input', fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    # 真实CT的AP投影（MIP）
    ax3 = plt.subplot(4, 6, 3)
    real_ap_mip = real.mean(0)  # 沿深度平均
    ax3.imshow(real_ap_mip, cmap='gray', vmin=0, vmax=1)
    ax3.set_title('Real MIP (AP)', fontsize=10)
    ax3.axis('off')
    
    # 生成CT的AP投影（MIP）
    ax4 = plt.subplot(4, 6, 4)
    fake_ap_mip = fake.mean(0)  # 沿深度平均
    ax4.imshow(fake_ap_mip, cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Generated MIP (AP)', fontsize=10, color='blue')
    ax4.axis('off')
    
    # 真实CT的LAT投影（MIP）
    ax5 = plt.subplot(4, 6, 5)
    real_lat_mip = real.mean(2)  # 沿宽度平均
    ax5.imshow(real_lat_mip, cmap='gray', vmin=0, vmax=1)
    ax5.set_title('Real MIP (LAT)', fontsize=10)
    ax5.axis('off')
    
    # 生成CT的LAT投影（MIP）
    ax6 = plt.subplot(4, 6, 6)
    fake_lat_mip = fake.mean(2)  # 沿宽度平均
    ax6.imshow(fake_lat_mip, cmap='gray', vmin=0, vmax=1)
    ax6.set_title('Generated MIP (LAT)', fontsize=10, color='blue')
    ax6.axis('off')
    
    # ========== 第二行：Axial切片（横断面） ==========
    for i in range(6):
        ax = plt.subplot(4, 6, 7 + i)
        slice_idx = int(D * (i + 1) / 7)
        
        if i < 3:
            # 真实的axial切片
            slice_data = real[slice_idx].numpy()
            title = f'Real Axial z={slice_idx}'
            color = 'black'
        else:
            # 生成的axial切片
            slice_data = fake[slice_idx].numpy()
            title = f'Gen Axial z={slice_idx}'
            color = 'blue'
        
        # 增强对比度
        slice_data = exposure.equalize_adapthist(slice_data, clip_limit=0.03)
        ax.imshow(slice_data, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')
    
    # ========== 第三行：Coronal切片（冠状面） ==========
    for i in range(6):
        ax = plt.subplot(4, 6, 13 + i)
        slice_idx = H // 2 + (i - 3) * 30  # 在中心附近取样
        slice_idx = max(0, min(H - 1, slice_idx))
        
        if i < 3:
            # 真实的coronal切片
            slice_data = real[:, slice_idx, :].numpy()
            title = f'Real Coronal y={slice_idx}'
            color = 'black'
        else:
            # 生成的coronal切片
            slice_data = fake[:, slice_idx, :].numpy()
            title = f'Gen Coronal y={slice_idx}'
            color = 'blue'
        
        # 增强对比度
        slice_data = exposure.equalize_adapthist(slice_data, clip_limit=0.03)
        ax.imshow(slice_data, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')
    
    # ========== 第四行：Sagittal切片（矢状面） ==========
    for i in range(6):
        ax = plt.subplot(4, 6, 19 + i)
        slice_idx = W // 2 + (i - 3) * 30  # 在中心附近取样
        slice_idx = max(0, min(W - 1, slice_idx))
        
        if i < 3:
            # 真实的sagittal切片
            slice_data = real[:, :, slice_idx].numpy()
            title = f'Real Sagittal x={slice_idx}'
            color = 'black'
        else:
            # 生成的sagittal切片
            slice_data = fake[:, :, slice_idx].numpy()
            title = f'Gen Sagittal x={slice_idx}'
            color = 'blue'
        
        # 增强对比度
        slice_data = exposure.equalize_adapthist(slice_data, clip_limit=0.03)
        ax.imshow(slice_data, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9, color=color)
        ax.axis('off')
    
    # 添加总标题
    title = f'Epoch {epoch}, Batch {batch_idx}'
    if metrics:
        title += f' | PSNR: {metrics["psnr"]:.2f} dB, SSIM: {metrics["ssim"]:.3f}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    # 添加图例说明
    fig.text(0.5, 0.02, 'Black titles = Real CT | Blue titles = Generated CT', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # 清理内存
    del real, fake, xray
    gc.collect()


def compute_metrics(ct_real, ct_fake):
    """Calculate evaluation metrics"""
    if ct_fake.shape != ct_real.shape:
        ct_fake = F.interpolate(ct_fake, size=ct_real.shape[2:], 
                               mode='trilinear', align_corners=False)
    
    real = (ct_real.detach().cpu().numpy() + 1) / 2
    fake = (ct_fake.detach().cpu().numpy() + 1) / 2
    
    psnr_values = []
    ssim_values = []
    
    for i in range(real.shape[0]):
        psnr_val = psnr(real[i, 0], fake[i, 0], data_range=1.0)
        psnr_values.append(psnr_val)
        
        ssim_slices = []
        for j in range(real.shape[2]):
            ssim_val = ssim(real[i, 0, j], fake[i, 0, j], data_range=1.0)
            ssim_slices.append(ssim_val)
        ssim_values.append(np.mean(ssim_slices))
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values)
    }


# ============================================
# Main Training Function
# ============================================

def main():
    """Main training function"""
    
    # Clear GPU cache first
    torch.cuda.empty_cache()
    gc.collect()
    
    # ========== Optimized Configuration ==========
    config = {
        # Data paths - UPDATE THESE TO YOUR PATHS
        'drr_dir': "/media/mldadmin/home/s125mdg35_04/CT RECONSTRACTION/DRR_Final",
        'ct_dir': "/media/mldadmin/home/s125mdg35_04/CT RECONSTRACTION/LIDC-IDRI",
        'output_dir': './outputs_transformer',
        'checkpoint_dir': './checkpoints_transformer',
        
        # Model parameters (adapted for memory)
        'base_channels': 24,      # Divisible by 8
        'disc_channels': 48,      # Divisible by 8
        'max_depth': 48,          # Fixed depth for stability
        'num_heads': 8,
        'num_transformer_blocks': 4,  # Reduced blocks
        'disc_layers': 3,
        'dropout': 0.1,
        
        # Key: Fix mode collapse loss weights
        'weight_gan': 0.1,       # Very low GAN weight!
        'weight_l1': 50.0,       # Very high L1 weight!
        'weight_proj': 1.0,       # Projection consistency
        
        # Key: Learning rate settings
        'lr_g': 0.00002,          # Higher generator learning rate
        'lr_d': 0.00001,        # Very low discriminator learning rate!
        'weight_decay': 0.001,
        
        # Key: Training strategy
        'd_update_freq': 3,      # Update discriminator every 10 steps only!
        
        # Training parameters
        'batch_size': 1,
        'epochs': 5000,
        'warmup_steps': 500,
        'grad_clip': 0.1,
        
        # System settings
        'use_amp': False,         # Mixed precision
        'device': 'cuda:2' if torch.cuda.is_available() else 'cpu',
        'num_workers': 0,        # Reduce memory usage
        'pin_memory': False,
        
        # Visualization
        'viz_freq': 50,
        'save_freq': 500,
        
        # Resume training
        'resume': True,  # Set to True to automatically resume from latest checkpoint
        'checkpoint_name': 'latest.pth',  # Which checkpoint to load
    }
    
    # Create directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Print configuration
    print("\n" + "="*70)
    print("TRANSFORMER GAN TRAINING WITH CHECKPOINT RESUME")
    print("="*70)
    print(f"Device: {config['device']}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(config['device'])}")
            mem = torch.cuda.get_device_properties(config['device']).total_memory / 1e9
            print(f"GPU Memory: {mem:.1f} GB")
        except:
            print("GPU info not available")
    print(f"Model: {config['base_channels']}ch base, {config['max_depth']} depth")
    print(f"Loss weights - GAN: {config['weight_gan']}, L1: {config['weight_l1']}, Proj: {config['weight_proj']}")
    print(f"Learning rates - G: {config['lr_g']}, D: {config['lr_d']}")
    print(f"Discriminator update frequency: every {config['d_update_freq']} steps")
    print(f"Resume training: {config['resume']}")
    print("="*70)
    
    # ========== Dataset ==========
    print("\nLoading dataset...")
    dataset = CTDataset(
        config['drr_dir'],
        config['ct_dir'],
        target_size=256,
        max_depth=config['max_depth'],
        augmentation=True,
        training=True
    )
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        collate_fn=collate_fn
    )
    
    print(f"✓ Dataset loaded: {len(train_set)} train, {len(val_set)} val")
    
    # ========== Trainer ==========
    trainer = ImprovedTransformerTrainer(config)
    
    # ========== Load Checkpoint if Resume ==========
    start_epoch = 0
    if config['resume']:
        checkpoint_path = os.path.join(config['checkpoint_dir'], config['checkpoint_name'])
        start_epoch = load_checkpoint(trainer, checkpoint_path, config['device'])
    else:
        print("\n🆕 Starting fresh training (resume=False)")
    
    # ========== Training Loop ==========
    print("\n" + "="*70)
    print(f"STARTING TRAINING FROM EPOCH {start_epoch + 1}")
    print("="*70)
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        trainer.G.train()
        trainer.D.train()
        
        epoch_metrics = {
            'D': [], 'G': [], 'L1': [], 'Proj': [],
            'PSNR': [], 'SSIM': []
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (xray, ct_real, pids, spacing) in enumerate(pbar):
            xray = xray.to(config['device'])
            ct_real = ct_real.to(config['device'])
            
            # Training step
            metrics = trainer.train_step(xray, ct_real)
            
            # Record metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])
            
            # Update progress bar
            pbar.set_postfix({
                'D': f"{metrics['D']:.3f}",
                'G': f"{metrics['G']:.3f}",
                'PSNR': f"{metrics['PSNR']:.1f}",
                'SSIM': f"{metrics['SSIM']:.2f}",
                'LR': f"{metrics['lr_g']:.0e}"
            })
            
            # Clear memory periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Visualization
            if batch_idx % config['viz_freq'] == 0:
                with torch.no_grad():
                    ct_fake = trainer.G(xray, ct_real.shape[2])
                    current_metrics = compute_metrics(ct_real, ct_fake)
                
                save_path = os.path.join(
                    config['output_dir'],
                    f'epoch_{epoch+1:03d}_batch_{batch_idx:04d}.png'
                )
                visualize_results(xray, ct_real, ct_fake, save_path,
                                epoch+1, batch_idx, current_metrics)
                
                # Print generated image statistics
                if batch_idx == 0:
                    print(f"\n  Generated CT stats:")
                    print(f"    Range: [{ct_fake.min():.3f}, {ct_fake.max():.3f}]")
                    print(f"    Mean: {ct_fake.mean():.3f}, Std: {ct_fake.std():.3f}")
                
                del ct_fake
            
            # Save checkpoint
            if trainer.step % config['save_freq'] == 0:
                ckpt_path = os.path.join(
                    config['checkpoint_dir'],
                    f'step_{trainer.step}.pth'
                )
                trainer.save_checkpoint(epoch, ckpt_path)
        
        # Epoch statistics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items() if len(v) > 0}
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Losses - D: {avg_metrics.get('D', 0):.4f}, G: {avg_metrics.get('G', 0):.4f}")
        print(f"  Losses - L1: {avg_metrics.get('L1', 0):.4f}, Proj: {avg_metrics.get('Proj', 0):.4f}")
        print(f"  Quality - PSNR: {avg_metrics.get('PSNR', 0):.2f} dB, SSIM: {avg_metrics.get('SSIM', 0):.3f}")
        
        # Check if adjustment is needed
        if epoch > 10 and avg_metrics.get('PSNR', 0) < 15:
            print("  ⚠ Warning: Low PSNR, consider increasing L1 weight")
        
        # Save best model
        current_psnr = avg_metrics.get('PSNR', 0)
        current_ssim = avg_metrics.get('SSIM', 0)
        
        if current_psnr > trainer.best_psnr:
            trainer.best_psnr = current_psnr
            best_path = os.path.join(config['checkpoint_dir'], 'best_psnr.pth')
            trainer.save_checkpoint(epoch, best_path)
            print(f"  🏆 New best PSNR: {trainer.best_psnr:.2f} dB")
        
        if current_ssim > trainer.best_ssim:
            trainer.best_ssim = current_ssim
            best_path = os.path.join(config['checkpoint_dir'], 'best_ssim.pth')
            trainer.save_checkpoint(epoch, best_path)
            print(f"  🏆 New best SSIM: {trainer.best_ssim:.3f}")
        
        # Save latest
        latest_path = os.path.join(config['checkpoint_dir'], 'latest.pth')
        trainer.save_checkpoint(epoch, latest_path)
        
        # Save periodically
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(config['checkpoint_dir'], f'epoch_{epoch+1}.pth')
            trainer.save_checkpoint(epoch, epoch_path)
        
        # Clean
        torch.cuda.empty_cache()
        gc.collect()
    
    # Training finished
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print(f"Best PSNR: {trainer.best_psnr:.2f} dB")
    print(f"Best SSIM: {trainer.best_ssim:.3f}")
    print("="*70)


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Run training
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
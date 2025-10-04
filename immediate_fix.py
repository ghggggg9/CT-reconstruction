#!/usr/bin/env python3
"""
修复checkpoint中的判别器通道不匹配问题
运行这个脚本来清理有问题的checkpoint
"""

import os
import torch
import shutil
from datetime import datetime

def fix_all_checkpoints():
    """修复所有checkpoint文件"""
    
    checkpoint_dir = './checkpoints_transformer'
    
    if not os.path.exists(checkpoint_dir):
        print(f"目录不存在: {checkpoint_dir}")
        return
    
    print("="*70)
    print("修复Checkpoint文件")
    print("="*70)
    
    # 创建备份目录
    backup_dir = f'./checkpoints_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(backup_dir, exist_ok=True)
    print(f"\n备份目录: {backup_dir}")
    
    # 获取所有.pth文件
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not pth_files:
        print("没有找到checkpoint文件")
        return
    
    print(f"\n找到 {len(pth_files)} 个checkpoint文件")
    print("-"*40)
    
    fixed_count = 0
    
    for filename in pth_files:
        filepath = os.path.join(checkpoint_dir, filename)
        backup_path = os.path.join(backup_dir, filename)
        
        print(f"\n处理: {filename}")
        
        try:
            # 备份原文件
            shutil.copy2(filepath, backup_path)
            print(f"  ✓ 已备份到: {backup_path}")
            
            # 加载checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # 检查是否包含判别器
            modified = False
            
            if 'D' in checkpoint:
                # 删除判别器状态
                del checkpoint['D']
                modified = True
                print(f"  ✓ 删除判别器权重")
            
            if 'opt_D' in checkpoint:
                # 删除判别器优化器状态
                del checkpoint['opt_D']
                modified = True
                print(f"  ✓ 删除判别器优化器")
            
            if modified:
                # 保存修改后的checkpoint
                torch.save(checkpoint, filepath)
                print(f"  ✓ 已保存修复后的文件")
                fixed_count += 1
                
                # 打印checkpoint信息
                if 'epoch' in checkpoint:
                    print(f"    Epoch: {checkpoint['epoch']}")
                if 'step' in checkpoint:
                    print(f"    Step: {checkpoint['step']}")
                if 'best_psnr' in checkpoint:
                    print(f"    Best PSNR: {checkpoint['best_psnr']:.2f}")
                if 'best_ssim' in checkpoint:
                    print(f"    Best SSIM: {checkpoint['best_ssim']:.3f}")
            else:
                print(f"  ℹ 不需要修改")
                
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            continue
    
    print("\n" + "="*70)
    print(f"修复完成!")
    print(f"  - 处理文件数: {len(pth_files)}")
    print(f"  - 修复文件数: {fixed_count}")
    print(f"  - 备份位置: {backup_dir}")
    print("="*70)
    
    print("\n下一步:")
    print("1. 运行训练脚本: python train.py")
    print("2. 判别器将自动重新初始化")
    print("3. 生成器将保留之前的训练进度")
    
    return fixed_count > 0


def verify_checkpoint(filepath):
    """验证单个checkpoint文件"""
    
    print(f"\n验证: {filepath}")
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        
        has_generator = 'G' in checkpoint
        has_discriminator = 'D' in checkpoint
        has_opt_g = 'opt_G' in checkpoint
        has_opt_d = 'opt_D' in checkpoint
        
        print(f"  生成器权重: {'✓' if has_generator else '✗'}")
        print(f"  判别器权重: {'✓' if has_discriminator else '✗'}")
        print(f"  生成器优化器: {'✓' if has_opt_g else '✗'}")
        print(f"  判别器优化器: {'✓' if has_opt_d else '✗'}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  配置:")
            print(f"    base_channels: {config.get('base_channels', 'N/A')}")
            print(f"    disc_channels: {config.get('disc_channels', 'N/A')}")
            print(f"    max_depth: {config.get('max_depth', 'N/A')}")
        
        return not has_discriminator  # 如果没有判别器，则验证通过
        
    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        return False


def clean_and_restart():
    """完全清理并重新开始（可选）"""
    
    print("\n" + "="*70)
    print("完全清理选项")
    print("="*70)
    
    response = input("\n是否要完全删除所有checkpoint并重新开始? (y/n): ").strip().lower()
    
    if response == 'y':
        checkpoint_dir = './checkpoints_transformer'
        
        if os.path.exists(checkpoint_dir):
            # 创建完整备份
            backup_dir = f'./checkpoints_full_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            shutil.copytree(checkpoint_dir, backup_dir)
            print(f"\n完整备份已创建: {backup_dir}")
            
            # 删除所有checkpoint
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pth'):
                    os.remove(os.path.join(checkpoint_dir, f))
            
            print("✓ 所有checkpoint已删除")
            print("\n现在可以运行 python train.py 开始全新训练")
        else:
            print(f"目录不存在: {checkpoint_dir}")
    else:
        print("取消操作")


def main():
    """主函数"""
    
    print("="*70)
    print("Checkpoint修复工具")
    print("="*70)
    print("\n选择操作:")
    print("1. 修复所有checkpoint（保留生成器，删除判别器）")
    print("2. 验证checkpoint状态")
    print("3. 完全清理并重新开始")
    print("4. 退出")
    
    choice = input("\n请选择 (1/2/3/4): ").strip()
    
    if choice == '1':
        success = fix_all_checkpoints()
        if success:
            print("\n✅ 修复成功! 现在可以运行 python train.py")
    
    elif choice == '2':
        checkpoint_dir = './checkpoints_transformer'
        if os.path.exists(checkpoint_dir):
            pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            for f in pth_files:
                verify_checkpoint(os.path.join(checkpoint_dir, f))
        else:
            print(f"目录不存在: {checkpoint_dir}")
    
    elif choice == '3':
        clean_and_restart()
    
    else:
        print("退出")


if __name__ == "__main__":
    main()
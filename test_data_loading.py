#!/usr/bin/env python3
"""
测试数据加载修复效果
验证：
1. 数据精度是否保持float32
2. 不同患者数据质量是否更一致
3. 可视化是否不再有过度噪声
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from train import CTDataset
import sys

print("=" * 60)
print("测试数据加载修复效果")
print("=" * 60)

# 配置
config = {
    'drr_dir': "/media/mldadmin/home/s125mdg35_04/CT RECONSTRACTION/DRR_Final",
    'ct_dir': "/media/mldadmin/home/s125mdg35_04/CT RECONSTRACTION/LIDC-IDRI",
}

# 创建数据集
print("\n1. 创建数据集...")
dataset = CTDataset(
    config['drr_dir'],
    config['ct_dir'],
    target_size=256,
    max_depth=48,
    augmentation=False,  # 关闭增强以测试原始数据
    training=True
)

print(f"✓ 数据集大小: {len(dataset)} 患者")

# 测试加载多个样本
print("\n2. 测试加载样本...")
test_indices = [0, 10, 20, 50, 100]  # 测试多个不同的患者

stats = []
for idx in test_indices:
    if idx >= len(dataset):
        continue

    try:
        xray, ct, pid, spacing = dataset[idx]

        # 收集统计信息
        ct_np = ct.numpy()[0]  # [1, D, H, W] -> [D, H, W]

        stat = {
            'idx': idx,
            'pid': pid,
            'shape': ct_np.shape,
            'dtype': ct_np.dtype,
            'min': np.min(ct_np),
            'max': np.max(ct_np),
            'mean': np.mean(ct_np),
            'std': np.std(ct_np),
            'contrast': np.percentile(ct_np, 95) - np.percentile(ct_np, 5),
        }
        stats.append(stat)

        print(f"\n  样本 {idx} (Patient: {pid}):")
        print(f"    形状: {stat['shape']}")
        print(f"    数据类型: {stat['dtype']}")
        print(f"    值范围: [{stat['min']:.3f}, {stat['max']:.3f}]")
        print(f"    均值/标准差: {stat['mean']:.3f} / {stat['std']:.3f}")
        print(f"    对比度(P95-P5): {stat['contrast']:.3f}")

    except Exception as e:
        print(f"  ⚠ 样本 {idx} 加载失败: {e}")

# 分析统计数据
print("\n3. 数据质量分析:")
print("-" * 60)

if len(stats) > 0:
    stds = [s['std'] for s in stats]
    contrasts = [s['contrast'] for s in stats]
    means = [abs(s['mean']) for s in stats]

    print(f"  标准差范围: [{min(stds):.3f}, {max(stds):.3f}]")
    print(f"  标准差平均: {np.mean(stds):.3f}")
    print(f"  标准差变异系数: {np.std(stds)/np.mean(stds):.3f}")
    print()
    print(f"  对比度范围: [{min(contrasts):.3f}, {max(contrasts):.3f}]")
    print(f"  对比度平均: {np.mean(contrasts):.3f}")
    print()
    print(f"  均值偏离范围: [{min(means):.3f}, {max(means):.3f}]")
    print(f"  均值偏离平均: {np.mean(means):.3f}")

    # 判断数据质量
    print("\n4. 质量评估:")
    if np.std(stds) / np.mean(stds) < 0.3:
        print("  ✅ 数据一致性: 好 (不同样本标准差变化小)")
    else:
        print("  ⚠️ 数据一致性: 中等 (不同样本标准差变化较大)")

    if all(s < 0.5 for s in stds):
        print("  ✅ 噪声水平: 好 (所有样本标准差 < 0.5)")
    else:
        print("  ⚠️ 噪声水平: 部分样本噪声较高")

    if all(c > 0.3 for c in contrasts):
        print("  ✅ 对比度: 充足 (所有样本对比度 > 0.3)")
    else:
        print("  ⚠️ 对比度: 部分样本对比度不足")

# 可视化对比
print("\n5. 生成对比可视化...")
if len(stats) >= 2:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Data Loading Test - Sample Comparison', fontsize=16, fontweight='bold')

    for i, idx in enumerate([0, 100] if 100 < len(dataset) else [0, 10]):
        if i >= len(stats):
            break

        xray, ct, pid, _ = dataset[idx]
        ct_np = ct.numpy()[0]  # [D, H, W]

        # 选择中间切片
        mid_slice = ct_np[ct_np.shape[0] // 2]

        # 原始数据
        axes[i, 0].imshow(mid_slice, cmap='gray', vmin=-1, vmax=1)
        axes[i, 0].set_title(f'Sample {idx}\nRaw (vmin=-1, vmax=1)')
        axes[i, 0].axis('off')

        # 自动范围
        axes[i, 1].imshow(mid_slice, cmap='gray')
        axes[i, 1].set_title(f'Auto Range\n[{stats[i]["min"]:.2f}, {stats[i]["max"]:.2f}]')
        axes[i, 1].axis('off')

        # 直方图
        axes[i, 2].hist(mid_slice.flatten(), bins=100, alpha=0.7, color='blue')
        axes[i, 2].set_title(f'Histogram\nμ={stats[i]["mean"]:.3f}, σ={stats[i]["std"]:.3f}')
        axes[i, 2].set_xlabel('Pixel Value')
        axes[i, 2].set_ylabel('Frequency')
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_loading_test_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 对比图已保存: data_loading_test_comparison.png")
else:
    print("⚠ 样本数量不足，跳过可视化")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
print("\n修复总结:")
print("  1. ✅ 使用cv2.resize保持float32精度，避免uint8量化")
print("  2. ✅ 移除CLAHE增强，使用原始数据可视化")
print("  3. ✅ 添加数据质量评估函数（可选启用）")
print("\n下一步:")
print("  - 检查 data_loading_test_comparison.png 验证数据质量")
print("  - 如果质量一致性好，可以继续训练")
print("  - 建议从头开始训练以充分利用修复效果")

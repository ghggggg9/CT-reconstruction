#!/usr/bin/env python3
import torch
from train import CTDataset, collate_fn
from torch.utils.data import DataLoader

def test_with_augmentation():
    """测试带增强的数据加载"""
    
    # 启用增强
    dataset = CTDataset(
        "/media/mldadmin/home/s125mdg35_04/CT RECONSTRACTION/DRR_Final",
        "/media/mldadmin/home/s125mdg35_04/CT RECONSTRACTION/LIDC-IDRI",
        target_size=256,
        max_depth=48,
        augmentation=True,  # 开启增强
        training=True
    )
    
    # 测试带shuffle的DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        collate_fn=collate_fn, 
        shuffle=True  # 开启shuffle
    )
    
    print("测试100个随机批次...")
    problem_count = 0
    
    for i, (xray_batch, ct_batch, pids, spacings) in enumerate(loader):
        if i >= 100:
            break
            
        if ct_batch.shape != (1, 1, 48, 256, 256):
            problem_count += 1
            print(f"❌ 批次 {i} ({pids[0]}):")
            print(f"   异常CT形状: {ct_batch.shape}")
            
            # 重新加载这个样本看看
            for j, pid in enumerate(dataset.patients):
                if pid == pids[0]:
                    xray, ct, _, _ = dataset[j]
                    print(f"   重新加载后: {ct.shape}")
                    break
    
    if problem_count == 0:
        print("✓ 100个批次全部正常!")
    else:
        print(f"❌ 发现 {problem_count} 个有问题的批次")

if __name__ == "__main__":
    test_with_augmentation()
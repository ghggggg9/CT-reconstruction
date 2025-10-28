import torch
import sys
sys.path.insert(0, '.')
from train import load_checkpoint, ImprovedTransformerGANTrainer

# 创建临时配置
config = {
    'base_channels': 24,
    'disc_channels': 48,
    'max_depth': 48,
    'num_heads': 8,
    'num_transformer_blocks': 4,
    'disc_layers': 3,
    'dropout': 0.1,
    'lr_g': 0.0001,
    'lr_d': 0.0001,
    'weight_decay': 0.0001,
    'weight_gan': 0.1,
    'weight_l1': 50.0,
    'weight_proj': 1.0,
    'use_amp': False,
    'device': 'cuda:2',
    'grad_clip': 1.0,
    'use_scheduler': True,
}

print('=== 测试Checkpoint加载 ===')
print('创建Trainer...')
trainer = ImprovedTransformerGANTrainer(config)
print('✓ Trainer创建成功')

print()
print('尝试加载checkpoint...')
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
load_checkpoint(trainer, './checkpoints_transformer/epoch_5010.pth', device)
print()
print('=== 测试完成 ===')

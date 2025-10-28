# 快速改进指南 - 立即行动版

**3个最重要的改进，预计提升PSNR 1-2 dB**

---

## 改进1: 调整Loss权重 ⚡⚡⚡

### 问题
L1权重过高(50.0) → 图像过于平滑，缺少细节

### 解决方案

**文件**: `train.py`
**行号**: 967-969

```python
# 修改前
'weight_l1': 50.0,
'weight_gan': 0.1,
'weight_proj': 1.0,

# 修改后
'weight_l1': 20.0,   # ← 减少2.5倍
'weight_gan': 0.5,   # ← 增加5倍
'weight_proj': 1.0,
```

### 预期效果
- ✅ 更多纹理细节
- ✅ 更锐利的边缘
- ✅ PSNR可能略降0.1-0.2 dB，但视觉质量明显提升
- ✅ SSIM预计持平或略升

### 风险
⚠️ 可能出现轻微伪影 (但通常可接受)

---

## 改进2: 增加Batch Size ⚡⚡⚡

### 问题
batch_size=1 → 训练极不稳定，PSNR波动 ±5 dB

### 解决方案

**方案A: 直接增加 (推荐)**

**文件**: `train.py`
**行号**: 996

```python
# 修改前
'batch_size': 1,

# 修改后
'batch_size': 2,  # 如果显存足够，可以用4
```

---

**方案B: 梯度累积 (显存不足时)**

**文件**: `train.py`
**位置**: 主训练循环 (约1100-1200行)

```python
# 1. 在config中添加
config = {
    # ... 其他配置 ...
    'gradient_accumulation_steps': 4,  # ← 新增
}

# 2. 修改训练循环
# 找到这段代码:
for batch_idx, (xray, ct_real, pids, spacing) in enumerate(pbar):

    # 在计算loss后，修改backward部分
    # 修改前:
    loss_G.backward()
    opt_G.step()
    opt_G.zero_grad()

    # 修改后:
    accumulation_steps = config.get('gradient_accumulation_steps', 1)
    loss_G = loss_G / accumulation_steps  # ← 缩放loss
    loss_G.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        # 梯度裁剪
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.G.parameters(),
                max_norm=config['grad_clip']
            )
        # 更新参数
        opt_G.step()
        opt_G.zero_grad()
```

### 预期效果
- ✅ PSNR/SSIM波动减少50%以上
- ✅ 训练曲线更平滑
- ✅ 收敛速度可能加快

### 风险
无风险 (梯度累积不增加显存占用)

---

## 改进3: 验证判别器权重已正确加载 ✅

### 问题
判别器权重未正确加载 → 5000轮训练成果丢失

### 解决方案

**状态**: ✅ **已在之前修改中完成**

验证方法:

```bash
# 重新运行训练，检查日志
python train.py

# 应该看到:
# → 预创建判别器first_layer...
# [Discriminator] 创建新的第一层: 97 -> 48
# ✓ first_layer已创建，准备加载权重
# ✓ 成功加载判别器权重  ← 这里应该成功，不再是"重新初始化"
```

如果看到 `✓ 成功加载判别器权重`，说明修复成功！

---

## 执行步骤

### Step 1: 备份当前代码
```bash
cd "/media/mldadmin/home/s125mdg35_04/CT RECONSTRACTION"
cp train.py train.py.backup
```

### Step 2: 应用改进1和2
按照上面的说明修改 `train.py`

### Step 3: 验证修改
```bash
# 快速语法检查
python -m py_compile train.py

# 应该没有错误输出
```

### Step 4: 开始训练 (从epoch 5000继续)
```bash
# 确保使用正确的checkpoint
python train.py

# 训练100个epoch作为快速验证
# 观察PSNR是否提升
```

### Step 5: 监控指标 (约2-3天后)
```bash
# 查看最新日志
tail -100 ./checkpoints_transformer/training.log

# 对比:
# - Epoch 5000 vs Epoch 5100 的PSNR
# - 波动是否减小
# - 视觉效果是否改善
```

---

## 成功标准

**100个epoch后 (约2-3天)**:

### 最低要求 (值得继续)
- PSNR平均提升 > 0.5 dB
- 单batch波动减小 > 30%

### 良好进展
- PSNR平均提升 > 1.0 dB
- SSIM提升 > 0.01
- 视觉上细节明显更多

### 优秀成果
- PSNR平均提升 > 1.5 dB
- SSIM提升 > 0.02
- 训练曲线非常平滑
- 生成图像接近真实CT

---

## 预期时间线

```
Day 1:    修改代码 (1小时)
Day 1-3:  训练100 epochs (约2-3天，取决于GPU)
Day 4:    分析结果，决定下一步

如果成功:
  → 继续训练500 epochs
  → 应用更多改进 (优先级2)

如果不明显:
  → 需要更深层次的架构改动
  → 查看完整分析报告的优先级2/3改进
```

---

## 回滚方案

如果改进后效果变差:

```bash
# 恢复原始代码
cp train.py.backup train.py

# 从epoch 5000重新开始
python train.py
```

---

## 常见问题

### Q1: 改了Loss权重后PSNR下降了?
**A**: 正常！前10-20个epoch可能略降，因为GAN需要重新平衡。继续训练50+ epochs再评估。

### Q2: Batch Size=2显存不够?
**A**: 使用梯度累积方案B，效果类似但不增加显存。

### Q3: 判别器加载还是失败?
**A**: 检查 `train.py` 第650-665行的修改是否正确应用。

---

## 联系信息

如需详细分析和长期优化方案，请查看:
📄 `Training_Analysis_Report_5000epochs.md`

---

**Good Luck! 🚀**

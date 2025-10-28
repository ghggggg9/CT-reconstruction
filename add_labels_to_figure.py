#!/usr/bin/env python3
"""
独立脚本：为已生成的可视化图片添加(a)-(x)标签
用于满足审稿人要求，在每个子图左上角添加字母标注

使用方法：
    python add_labels_to_figure.py epoch_4500_batch_0000.png

输出：
    epoch_4500_batch_0000_labeled.png
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
import os


def add_labels_to_visualization(input_path, output_path=None):
    """
    为4x6布局的可视化图片添加(a)-(x)标签

    参数：
        input_path: 输入图片路径
        output_path: 输出图片路径（如果为None，自动生成）
    """

    # 如果未指定输出路径，自动生成
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_labeled{ext}"

    # 读取图片
    print(f"📖 正在读取图片: {input_path}")
    img = Image.open(input_path)
    img_array = np.array(img)

    # 获取图片尺寸
    height, width = img_array.shape[:2]
    print(f"📐 图片尺寸: {width} x {height}")

    # 创建可绘制对象
    draw = ImageDraw.Draw(img)

    # 设置字体（尝试多个字体选项）
    font_size = 40  # 字体大小
    font = None

    # 尝试加载系统字体
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:\\Windows\\Fonts\\arialbd.ttf",  # Windows
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                print(f"✓ 使用字体: {font_path}")
                break
            except:
                continue

    # 如果找不到字体，使用默认字体
    if font is None:
        font = ImageFont.load_default()
        print("⚠ 使用默认字体（质量可能较低）")

    # 计算子图布局（4行 x 6列）
    rows = 4
    cols = 6
    total_subplots = rows * cols  # 24个子图

    # 估算每个子图的尺寸
    # 考虑到有标题栏和底部说明，实际绘图区域需要调整
    title_height = int(height * 0.05)  # 顶部标题约占5%
    bottom_margin = int(height * 0.03)  # 底部说明约占3%

    plot_height = height - title_height - bottom_margin
    plot_width = width

    subplot_height = plot_height // rows
    subplot_width = plot_width // cols

    print(f"📊 子图布局: {rows}行 x {cols}列")
    print(f"📏 每个子图尺寸约: {subplot_width} x {subplot_height}")

    # 生成标签 (a) 到 (x)
    labels = [f"({chr(97 + i)})" for i in range(total_subplots)]  # 97 = 'a'

    # 标签位置偏移（从子图左上角的偏移）
    label_offset_x = 10  # 距离左边界10像素
    label_offset_y = 10  # 距离上边界10像素

    # 为每个子图添加标签
    label_idx = 0
    for row in range(rows):
        for col in range(cols):
            if label_idx >= total_subplots:
                break

            # 计算子图左上角位置
            x = col * subplot_width
            y = title_height + row * subplot_height

            # 标签文本位置
            label_x = x + label_offset_x
            label_y = y + label_offset_y

            label_text = labels[label_idx]

            # 获取文本边界框（用于绘制背景）
            bbox = draw.textbbox((label_x, label_y), label_text, font=font)

            # 扩展背景框（增加padding）
            padding = 8
            bg_box = [
                bbox[0] - padding,
                bbox[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding
            ]

            # 绘制白色圆角背景框
            draw.rounded_rectangle(bg_box, radius=8, fill='white', outline='black', width=2)

            # 绘制黑色粗体标签文本
            draw.text((label_x, label_y), label_text, fill='black', font=font)

            print(f"  ✓ 添加标签 {label_text} 在位置 ({label_x}, {label_y})")

            label_idx += 1

    # 保存带标签的图片
    img.save(output_path, quality=95, dpi=(300, 300))
    print(f"\n✅ 成功保存带标签图片: {output_path}")
    print(f"📁 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return output_path


def print_figure_caption():
    """打印改进后的Figure Caption"""

    print("\n" + "="*80)
    print("📝 改进后的 Figure Caption（用于论文）")
    print("="*80)

    caption_en = """
Figure X: Qualitative evaluation of 3D CT reconstruction at epoch 4500.
(a-b) Input biplanar X-rays in anterior-posterior (AP) and lateral (LAT) views.
(c-f) Maximum intensity projections (MIP) of real and generated CT volumes for
projection consistency validation, shown in both AP and LAT orientations.
(g-l) Axial slices displaying cross-sectional anatomy at three depth levels,
with (g-i) from ground truth CT and (j-l) from generated CT.
(m-r) Coronal slices revealing anterior-posterior anatomical structures,
with (m-o) showing ground truth and (p-r) showing generated volumes.
(s-x) Sagittal slices demonstrating lateral anatomy, with (s-u) from ground
truth and (v-x) from generated CT. Quantitative metrics: PSNR=23.85 dB, SSIM=0.773.
"""

    caption_cn = """
图X：第4500轮训练的3D CT重建定性评估。
(a-b) 前后位(AP)和侧位(LAT)双视角输入X光片。
(c-f) 真实与生成CT体积的最大密度投影(MIP)，用于投影一致性验证，分别展示AP和LAT方向。
(g-l) 三个深度层级的横断面切片，其中(g-i)为真实CT，(j-l)为生成CT。
(m-r) 冠状面切片展示前后向解剖结构，(m-o)为真实CT，(p-r)为生成CT。
(s-x) 矢状面切片展示侧向解剖结构，(s-u)为真实CT，(v-x)为生成CT。
定量指标：PSNR=23.85 dB，SSIM=0.773。
"""

    print("\n【英文版】")
    print(caption_en)

    print("\n【中文版】")
    print(caption_cn)

    print("="*80)


def print_label_mapping():
    """打印标签映射表"""

    print("\n" + "="*80)
    print("🏷️  标签映射表（Label Mapping）")
    print("="*80)

    mapping = {
        "Row 1 - Inputs & Projections": [
            ("(a)", "AP X-ray Input", "前后位X光输入"),
            ("(b)", "LAT X-ray Input", "侧位X光输入"),
            ("(c)", "Real MIP (AP)", "真实CT前后位投影"),
            ("(d)", "Generated MIP (AP)", "生成CT前后位投影"),
            ("(e)", "Real MIP (LAT)", "真实CT侧位投影"),
            ("(f)", "Generated MIP (LAT)", "生成CT侧位投影"),
        ],
        "Row 2 - Axial Slices": [
            ("(g)", "Real Axial z=6", "真实横断面z=6"),
            ("(h)", "Real Axial z=13", "真实横断面z=13"),
            ("(i)", "Real Axial z=20", "真实横断面z=20"),
            ("(j)", "Generated Axial z=27", "生成横断面z=27"),
            ("(k)", "Generated Axial z=34", "生成横断面z=34"),
            ("(l)", "Generated Axial z=41", "生成横断面z=41"),
        ],
        "Row 3 - Coronal Slices": [
            ("(m)", "Real Coronal y=38", "真实冠状面y=38"),
            ("(n)", "Real Coronal y=68", "真实冠状面y=68"),
            ("(o)", "Real Coronal y=98", "真实冠状面y=98"),
            ("(p)", "Generated Coronal y=128", "生成冠状面y=128"),
            ("(q)", "Generated Coronal y=158", "生成冠状面y=158"),
            ("(r)", "Generated Coronal y=188", "生成冠状面y=188"),
        ],
        "Row 4 - Sagittal Slices": [
            ("(s)", "Real Sagittal x=38", "真实矢状面x=38"),
            ("(t)", "Real Sagittal x=68", "真实矢状面x=68"),
            ("(u)", "Real Sagittal x=98", "真实矢状面x=98"),
            ("(v)", "Generated Sagittal x=128", "生成矢状面x=128"),
            ("(w)", "Generated Sagittal x=158", "生成矢状面x=158"),
            ("(x)", "Generated Sagittal x=188", "生成矢状面x=188"),
        ]
    }

    for section, items in mapping.items():
        print(f"\n{section}:")
        print("-" * 80)
        for label, en, cn in items:
            print(f"  {label:4s} | {en:30s} | {cn}")

    print("\n" + "="*80)


def main():
    """主函数"""

    print("\n" + "="*80)
    print("🏷️  Figure Label Adder - 为可视化图片添加标签")
    print("="*80)

    # 检查命令行参数
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python add_labels_to_figure.py <输入图片路径> [输出图片路径]")
        print("\n示例:")
        print("  python add_labels_to_figure.py epoch_4500_batch_0000.png")
        print("  python add_labels_to_figure.py epoch_4500_batch_0000.png output_labeled.png")

        # 尝试使用默认文件
        default_file = "epoch_4500_batch_0000.png"
        if os.path.exists(default_file):
            print(f"\n找到默认文件: {default_file}")
            print("使用默认文件进行处理...")
            input_path = default_file
        else:
            print(f"\n❌ 未找到默认文件: {default_file}")
            return
    else:
        input_path = sys.argv[1]

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"\n❌ 错误: 文件不存在: {input_path}")
        return

    # 获取输出路径
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # 处理图片
    try:
        result_path = add_labels_to_visualization(input_path, output_path)

        # 打印标签映射表
        print_label_mapping()

        # 打印Figure Caption
        print_figure_caption()

        print("\n" + "="*80)
        print("✅ 处理完成！")
        print("="*80)
        print(f"\n原始图片: {input_path}")
        print(f"标注图片: {result_path}")
        print("\n现在你可以:")
        print("  1. 查看生成的带标签图片")
        print("  2. 复制上面的 Figure Caption 用于论文")
        print("  3. 参考标签映射表在文中引用具体子图")

    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

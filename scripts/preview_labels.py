"""
在 images 上绘制 YOLO 标注，生成预览图
输出 preview/ 下 100 张带框图像，便于检查标注
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 类别名（与 render_and_export 一致）
CLASS_NAMES = [
    "snake_head_green", "snake_head_blue", "snake_head_orange",
    "snake_body_green", "snake_body_blue", "snake_body_orange",
    "food_green", "food_blue", "food_orange",
    "x2_green", "x2_blue", "x2_orange",
]

# 框颜色 (R,G,B)，按类别区分
BOX_COLORS = [
    (76, 175, 80), (33, 150, 243), (255, 152, 0),   # 蛇头 绿蓝橙
    (56, 142, 60), (25, 118, 210), (230, 81, 0),    # 蛇身
    (129, 199, 132), (100, 181, 246), (255, 183, 77),  # 食物
    (46, 125, 50), (13, 71, 161), (230, 81, 0),     # x2
]


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """YOLO 归一化 -> 像素 xyxy"""
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_preview(img_path: Path, lbl_path: Path, out_path: Path, img_w: int = 600, img_h: int = 600) -> None:
    """在图像上绘制标注并保存"""
    try:
        from PIL import Image
        import PIL.ImageDraw as ImageDraw
    except ImportError:
        print("请安装 Pillow: pip install Pillow")
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            color = BOX_COLORS[c % len(BOX_COLORS)]
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 12), name, fill=color)

    img.save(out_path)
    img.close()


def main():
    import argparse
    p = argparse.ArgumentParser(description="绘制 YOLO 标注预览")
    p.add_argument("--dataset", "-d", default=None, help="数据集目录，默认 dataset/")
    p.add_argument("--split", "-s", default="train", choices=["train", "val"], help="用 train 或 val")
    p.add_argument("--num", "-n", type=int, default=100, help="生成预览数量")
    p.add_argument("--output", "-o", default="preview", help="预览输出目录")
    args = p.parse_args()

    dataset = Path(args.dataset or ROOT / "dataset")
    img_dir = dataset / args.split / "images"
    lbl_dir = dataset / args.split / "labels"
    out_dir = Path(args.output or ROOT / "preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        print(f"请先运行 render_and_export.py 生成数据集")
        print(f"  python scripts/render_and_export.py")
        sys.exit(1)

    images = sorted(img_dir.glob("*.png"))[: args.num]
    if not images:
        print(f"未找到图像: {img_dir}")
        sys.exit(1)

    print(f"生成 {len(images)} 张预览 -> {out_dir}")
    for i, img_path in enumerate(images):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        out_path = out_dir / f"preview_{i:04d}_{img_path.name}"
        draw_preview(img_path, lbl_path, out_path)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(images)}")

    print(f"完成: {out_dir}")
    try:
        import webbrowser
        import os
        webbrowser.open(f"file://{out_dir.absolute()}")
    except Exception:
        print("请手动打开 preview/ 目录查看")


if __name__ == "__main__":
    main()

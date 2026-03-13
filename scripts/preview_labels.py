"""
在 images 上绘制 YOLO 标注 + 行为正确性标签
输出 preview/ 下 100 张带框图像，便于检查标注
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 类别名（与 render_and_export 一致，不区分颜色）
CLASS_NAMES = ["snake_head", "snake_body", "food", "x2"]

# 框颜色 (R,G,B)：蛇头用深色、蛇身用浅色，与渲染一致
BOX_COLORS = [
    (46, 125, 50), (76, 175, 80),   # snake_head(深绿), snake_body(浅绿)
    (255, 183, 77), (255, 193, 7),  # food, x2
]

# 行为标签颜色
REASON_NAMES = {
    "ate_x2_then_food": "先吃x2再吃食物(对)",
    "ate_food_no_x2": "吃食物(对)",
    "in_progress": "进行中",
    "self_collision": "撞到自己(错)",
    "snake_collision": "蛇间碰撞(错)",
    "x2_wasted": "x2浪费(错)",
    "timeout": "超时(错)",
}
# 每蛇颜色（与 game.SNAKE_COLORS body 一致）
SNAKE_COLORS = [(76, 175, 80), (33, 150, 243), (255, 152, 0)]  # 绿蓝橙
INCORRECT_COLOR = (244, 67, 54)  # 错误时用红色


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """YOLO 归一化 -> 像素 xyxy"""
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_preview(
    img_path: Path,
    lbl_path: Path,
    beh_path: Path | None,
    out_path: Path,
    img_w: int = 640,
    img_h: int = 640,
) -> None:
    """在图像上绘制 YOLO 标注 + 行为标签并保存"""
    try:
        from PIL import Image
        import PIL.ImageDraw as ImageDraw
        import PIL.ImageFont as ImageFont
    except ImportError:
        print("请安装 Pillow: pip install Pillow")
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for fn in ("msyh.ttc", "simhei.ttf", "arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        try:
            font = ImageFont.truetype(fn, 14)
            break
        except Exception:
            continue

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
            text_y = max(0, y1 - 12)
            draw.text((x1, text_y), name, fill=color, font=font)

    # 绘制行为正确性标签（底部扩展面板，不覆盖游戏区）
    if beh_path and beh_path.exists():
        try:
            data = json.loads(beh_path.read_text(encoding="utf-8"))
            anns = data.get("snake_annotations", [])
        except Exception:
            anns = []
        if anns:
            line_h = 20
            panel_h = len(anns) * line_h + 16
            new_h = img_h + panel_h
            out_img = Image.new("RGB", (img_w, new_h), (28, 28, 32))
            out_img.paste(img, (0, 0))
            draw = ImageDraw.Draw(out_img)
            y = img_h + 8
            for i, ann in enumerate(anns):
                lbl = ann.get("label", "correct")
                rsn = ann.get("reason", "in_progress")
                rsn_zh = REASON_NAMES.get(rsn, rsn)
                txt = f"蛇{i+1}: {'正确' if lbl == 'correct' else '错误'} - {rsn_zh}"
                snake_color = SNAKE_COLORS[i % len(SNAKE_COLORS)]
                color = snake_color if lbl == "correct" else INCORRECT_COLOR
                draw.text((12, y), txt, fill=color, font=font)
                y += line_h
            img = out_img

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
    beh_dir = dataset / args.split / "behavior"
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
        beh_path = beh_dir / (img_path.stem + ".json") if beh_dir.exists() else None
        out_path = out_dir / f"preview_{i:04d}_{img_path.name}"
        draw_preview(img_path, lbl_path, beh_path, out_path)
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

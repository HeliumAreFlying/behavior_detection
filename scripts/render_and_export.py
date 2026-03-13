"""
从 batch JSON 渲染游戏画面，生成 YOLO 格式数据集
输出: dataset/train, dataset/val (images + labels)
"""

import json
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pygame
except ImportError:
    print("请先安装 pygame: pip install pygame")
    sys.exit(1)

from game import SNAKE_COLORS

# 渲染参数（与 replay_ui 一致）
CELL_SIZE = 40
GRID_W = 15
GRID_H = 15
IMG_W = GRID_W * CELL_SIZE
IMG_H = GRID_H * CELL_SIZE
BG = (28, 28, 32)
GRID_LINE = (48, 48, 56)

# YOLO 类别: 不区分颜色，由跟踪 (ByteTrack) 区分不同目标
CLASS_NAMES = ["snake_head", "snake_body", "food", "x2"]


def grid_to_yolo(gx: int, gy: int) -> tuple[float, float, float, float]:
    """网格坐标转 YOLO 归一化 (x_center, y_center, w, h)"""
    gx, gy = gx % GRID_W, gy % GRID_H
    xc = (gx * CELL_SIZE + CELL_SIZE / 2) / IMG_W
    yc = (gy * CELL_SIZE + CELL_SIZE / 2) / IMG_H
    w = CELL_SIZE / IMG_W
    h = CELL_SIZE / IMG_H
    return xc, yc, w, h


def scene_to_bboxes(scene: dict) -> list[tuple[int, float, float, float, float]]:
    """从 scene 提取 YOLO 标注 (class_id, xc, yc, w, h)"""
    lines: list[tuple[int, float, float, float, float]] = []
    snakes_data = scene.get("snakes")
    if not snakes_data:
        return lines

    for s in snakes_data:
        body = s.get("body", [])
        food = s.get("food", [0, 0])
        x2 = s.get("x2")

        for i, pos in enumerate(body):
            try:
                gx, gy = int(pos[0]), int(pos[1])
            except (IndexError, TypeError):
                continue
            gx = ((gx % GRID_W) + GRID_W) % GRID_W
            gy = ((gy % GRID_H) + GRID_H) % GRID_H
            xc, yc, w, h = grid_to_yolo(gx, gy)
            cls = 0 if i == 0 else 1  # snake_head / snake_body
            lines.append((cls, xc, yc, w, h))

        if food:
            fx = ((int(food[0]) % GRID_W) + GRID_W) % GRID_W
            fy = ((int(food[1]) % GRID_H) + GRID_H) % GRID_H
            xc, yc, w, h = grid_to_yolo(fx, fy)
            lines.append((2, xc, yc, w, h))  # food
        if x2:
            xx = ((int(x2[0]) % GRID_W) + GRID_W) % GRID_W
            xy = ((int(x2[1]) % GRID_H) + GRID_H) % GRID_H
            xc, yc, w, h = grid_to_yolo(xx, xy)
            lines.append((3, xc, yc, w, h))  # x2

    return lines


def render_scene(scene: dict) -> "pygame.Surface":
    """将 scene 渲染为 600x600 图像（仅游戏网格）"""
    screen = pygame.Surface((IMG_W, IMG_H))
    screen.fill(BG)

    snakes_data = scene.get("snakes")
    if not snakes_data:
        return screen

    # 网格线
    for x in range(GRID_W + 1):
        pygame.draw.line(screen, GRID_LINE, (x * CELL_SIZE, 0), (x * CELL_SIZE, IMG_H))
    for y in range(GRID_H + 1):
        pygame.draw.line(screen, GRID_LINE, (0, y * CELL_SIZE), (IMG_W, y * CELL_SIZE))

    # 食物(F)、x2、蛇身(H/无字)，每阵营一色
    try:
        cell_font = pygame.font.SysFont("arial", 14)
    except Exception:
        cell_font = None
    for s in snakes_data:
        cid = s.get("color_id", 0) % len(SNAKE_COLORS)
        col = SNAKE_COLORS[cid]["body"]
        food = s.get("food", [0, 0])
        x2 = s.get("x2")
        if food:
            fx, fy = food[0] % GRID_W, food[1] % GRID_H
            rect = (fx * CELL_SIZE + 2, fy * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(screen, col, rect, border_radius=4)
            if cell_font:
                txt = cell_font.render("F", True, (40, 40, 40))
                tw, th = txt.get_size()
                screen.blit(txt, (fx * CELL_SIZE + (CELL_SIZE - tw) // 2, fy * CELL_SIZE + (CELL_SIZE - th) // 2))
        if x2:
            xx, xy = x2[0] % GRID_W, x2[1] % GRID_H
            rect = (xx * CELL_SIZE + 2, xy * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(screen, col, rect, border_radius=4)
            if cell_font:
                txt = cell_font.render("x2", True, (40, 40, 40))
                tw, th = txt.get_size()
                screen.blit(txt, (xx * CELL_SIZE + (CELL_SIZE - tw) // 2, xy * CELL_SIZE + (CELL_SIZE - th) // 2))

    # 蛇身：同色，蛇头显示 H
    for s in snakes_data:
        body = s.get("body", [])
        cid = s.get("color_id", 0) % len(SNAKE_COLORS)
        col = SNAKE_COLORS[cid]["body"]
        for i, pos in enumerate(body):
            try:
                sx, sy = int(pos[0]), int(pos[1])
            except (IndexError, TypeError):
                continue
            sx = ((sx % GRID_W) + GRID_W) % GRID_W
            sy = ((sy % GRID_H) + GRID_H) % GRID_H
            rect = (sx * CELL_SIZE + 1, sy * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2)
            pygame.draw.rect(screen, col, rect, border_radius=3)
            if i == 0 and cell_font:
                txt = cell_font.render("H", True, (40, 40, 40))
                tw, th = txt.get_size()
                screen.blit(txt, (sx * CELL_SIZE + (CELL_SIZE - tw) // 2, sy * CELL_SIZE + (CELL_SIZE - th) // 2))

    return screen


def is_key_frame(scene: dict, prev_scene: dict | None, is_last: bool, ep_reason: str) -> bool:
    """判断是否为关键帧：吃到食物/x2、超时刷新、碰撞等"""
    if prev_scene is None:
        return True  # 每局首帧保留
    if is_last and ep_reason in ("self_collision", "snake_collision"):
        return True  # 碰撞前一帧
    snakes_cur = scene.get("snakes", [])
    snakes_prev = prev_scene.get("snakes", [])
    for i in range(min(len(snakes_cur), len(snakes_prev))):
        sc, sp = snakes_cur[i], snakes_prev[i]
        # 吃到食物：得分增加
        if sc.get("score", 0) > sp.get("score", 0):
            return True
        # 吃到 x2：x2 从有变无（且得分未变，排除 ate_food_x2）
        if sp.get("x2") is not None and sc.get("x2") is None:
            if sc.get("score", 0) == sp.get("score", 0):
                return True
        # 超时：食物位置变了但得分未变
        if sc.get("food") != sp.get("food") and sc.get("score", 0) == sp.get("score", 0):
            return True
    return False


def main():
    import argparse
    import random
    p = argparse.ArgumentParser(description="渲染 batch JSON 为 YOLO 数据集")
    p.add_argument("--batches", "-b", default=None, help="batch 目录，默认 batches/")
    p.add_argument("--output", "-o", default="dataset", help="输出根目录")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    p.add_argument("--skip-n", type=int, default=5, help="非关键帧每 N 帧采样 1 帧，1 表示不跳帧")
    args = p.parse_args()

    batches_dir = Path(args.batches or ROOT / "batches")
    output_dir = Path(args.output or ROOT / "dataset")
    skip_n = max(1, args.skip_n)  # 0 表示不跳帧

    batch_files = sorted(batches_dir.glob("batch_*.json"))
    if not batch_files:
        print(f"未找到 batch 文件: {batches_dir}/batch_*.json")
        sys.exit(1)

    train_img = output_dir / "train" / "images"
    train_lbl = output_dir / "train" / "labels"
    train_beh = output_dir / "train" / "behavior"
    val_img = output_dir / "val" / "images"
    val_lbl = output_dir / "val" / "labels"
    val_beh = output_dir / "val" / "behavior"
    for d in (train_img, train_lbl, train_beh, val_img, val_lbl, val_beh):
        d.mkdir(parents=True, exist_ok=True)

    pygame.init()
    pygame.display.set_mode((1, 1))

    all_samples: list[tuple[dict, int, int, str]] = []
    for bf in batch_files:
        data = json.loads(bf.read_text(encoding="utf-8"))
        for ep_idx, ep in enumerate(data.get("episodes", [])):
            scenes = ep.get("scenes", [])
            ep_reason = ep.get("reason", "")
            for sc_idx, scene in enumerate(scenes):
                if not scene.get("snakes"):
                    continue
                prev = scenes[sc_idx - 1] if sc_idx > 0 else None
                is_last = sc_idx == len(scenes) - 1
                key = is_key_frame(scene, prev, is_last, ep_reason)
                if key or (skip_n <= 1) or (sc_idx % skip_n == 0):
                    all_samples.append((scene, ep_idx, sc_idx, bf.name))

    n = len(all_samples)
    val_n = int(n * args.val_ratio)
    train_n = n - val_n
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_idx = set(indices[:train_n])
    val_idx = set(indices[train_n:])

    metadata_list: list[dict] = []
    global_idx = 0
    for i, (scene, ep_idx, sc_idx, batch_name) in enumerate(all_samples):
        split = "val" if i in val_idx else "train"
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"
        beh_dir = output_dir / split / "behavior"
        name = f"{global_idx:06d}"
        img_path = img_dir / f"{name}.png"
        lbl_path = lbl_dir / f"{name}.txt"
        beh_path = beh_dir / f"{name}.json"

        surf = render_scene(scene)
        pygame.image.save(surf, img_path)

        bboxes = scene_to_bboxes(scene)
        lbl_lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in bboxes]
        lbl_path.write_text("\n".join(lbl_lines), encoding="utf-8")

        # 行为正确性标签：每蛇的 label + reason（针对当前食物）
        snake_ann = scene.get("snake_annotations", [])
        beh_data = {"snake_annotations": snake_ann}
        beh_path.write_text(json.dumps(beh_data, ensure_ascii=False), encoding="utf-8")

        metadata_list.append({
            "id": name,
            "split": split,
            "batch": batch_name,
            "episode": ep_idx,
            "scene": sc_idx,
        })
        global_idx += 1
        if global_idx % 500 == 0:
            print(f"已处理 {global_idx}/{n} ...")

    # data.yaml for YOLO
    abs_path = output_dir.resolve()
    data_yaml = f"""# YOLO 数据集配置
path: {abs_path}
train: train/images
val: val/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # 元数据：供跟踪脚本按 episode 分组，重建时序
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"完成: train {train_n}, val {val_n}, 总计 {n} (跳帧 skip_n={skip_n})")
    print(f"输出: {output_dir}")
    print(f"配置: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

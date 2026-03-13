"""
从 batch JSON 渲染游戏画面，生成 YOLO 格式数据集
输出: dataset/train, dataset/val (images + labels)
"""

import json
import os
import sys
import warnings
from pathlib import Path

# 无头环境下抑制 pygame/SDL 的 ALSA 音频警告（不影响渲染）
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# 抑制 pkg_resources 弃用警告（来自 pygame 内部）
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pygame
except ImportError:
    print("请先安装 pygame: pip install pygame")
    sys.exit(1)

from game import SNAKE_COLORS

# 渲染参数（与 replay_ui 一致），640x640 输出
IMG_W = IMG_H = 640
GRID_W = GRID_H = 15
CELL_SIZE = IMG_W / GRID_W  # 640/15 ≈ 42.67
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


def _draw_snake_head(screen: "pygame.Surface", sx: int, sy: int, col: tuple,
                     shape: str, dx: int = 1, dy: int = 0) -> None:
    """蛇头形状：diamond 开局 | triangle 前进(尖指方向) | circle 撞击死亡"""
    cx = int(sx * CELL_SIZE + CELL_SIZE / 2)
    cy = int(sy * CELL_SIZE + CELL_SIZE / 2)
    r = max(2, int(CELL_SIZE / 2) - 2)
    if shape == "diamond":
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        pygame.draw.polygon(screen, col, pts)
        pygame.draw.polygon(screen, (40, 40, 40), pts, 1)
    elif shape == "circle":
        pygame.draw.circle(screen, col, (cx, cy), r)
        pygame.draw.circle(screen, (40, 40, 40), (cx, cy), r, 1)
    else:  # triangle, tip in (dx, dy) direction
        if dx == 0 and dy == 0:
            dx, dy = 1, 0
        tip = (cx + dx * r, cy + dy * r)
        perp = (-dy, dx) if (dx or dy) else (1, 0)
        base1 = (cx - dx * r * 0.6 + perp[0] * r * 0.6, cy - dy * r * 0.6 + perp[1] * r * 0.6)
        base2 = (cx - dx * r * 0.6 - perp[0] * r * 0.6, cy - dy * r * 0.6 - perp[1] * r * 0.6)
        pts = [tip, base1, base2]
        pygame.draw.polygon(screen, col, pts)
        pygame.draw.polygon(screen, (40, 40, 40), pts, 1)


def render_scene(scene: dict, scene_idx: int = 0, prev_scene: dict | None = None,
                 total_scenes: int = 1) -> "pygame.Surface":
    """将 scene 渲染为 640x640 图像。蛇头：开局菱形→三角形(尖指方向)→撞击圆形"""
    screen = pygame.Surface((IMG_W, IMG_H))
    screen.fill(BG)

    snakes_data = scene.get("snakes")
    if not snakes_data:
        return screen

    anns = scene.get("snake_annotations", [])
    is_last = (scene_idx == total_scenes - 1) if total_scenes else False
    dead_reasons = {"self_collision", "snake_collision"}

    # 网格线
    for x in range(GRID_W + 1):
        px = int(x * CELL_SIZE)
        pygame.draw.line(screen, GRID_LINE, (px, 0), (px, IMG_H))
    for y in range(GRID_H + 1):
        py = int(y * CELL_SIZE)
        pygame.draw.line(screen, GRID_LINE, (0, py), (IMG_W, py))

    # 食物(F)、x2
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
            rect = (int(fx * CELL_SIZE + 2), int(fy * CELL_SIZE + 2), int(CELL_SIZE - 4), int(CELL_SIZE - 4))
            pygame.draw.rect(screen, col, rect, border_radius=4)
            if cell_font:
                txt = cell_font.render("F", True, (40, 40, 40))
                tw, th = txt.get_size()
                screen.blit(txt, (int(fx * CELL_SIZE + (CELL_SIZE - tw) // 2), int(fy * CELL_SIZE + (CELL_SIZE - th) // 2)))
        if x2:
            xx, xy = x2[0] % GRID_W, x2[1] % GRID_H
            rect = (int(xx * CELL_SIZE + 2), int(xy * CELL_SIZE + 2), int(CELL_SIZE - 4), int(CELL_SIZE - 4))
            pygame.draw.rect(screen, col, rect, border_radius=4)
            if cell_font:
                txt = cell_font.render("x2", True, (40, 40, 40))
                tw, th = txt.get_size()
                screen.blit(txt, (int(xx * CELL_SIZE + (CELL_SIZE - tw) // 2), int(xy * CELL_SIZE + (CELL_SIZE - th) // 2)))

    # 蛇身：蛇头按状态画菱形/三角/圆
    prev_heads = []
    if prev_scene and prev_scene.get("snakes"):
        for sp in prev_scene["snakes"]:
            b = sp.get("body", [])
            if b:
                prev_heads.append((int(b[0][0]) % GRID_W, int(b[0][1]) % GRID_H))
            else:
                prev_heads.append(None)

    for si, s in enumerate(snakes_data):
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
            rect = (int(sx * CELL_SIZE + 1), int(sy * CELL_SIZE + 1), int(CELL_SIZE - 2), int(CELL_SIZE - 2))
            pygame.draw.rect(screen, col, rect, border_radius=3)
            if i == 0:
                ann = anns[si] if si < len(anns) else {}
                reason = ann.get("reason", "")
                if scene_idx == 0:
                    _draw_snake_head(screen, sx, sy, col, "diamond")
                elif is_last and reason in dead_reasons:
                    _draw_snake_head(screen, sx, sy, col, "circle")
                else:
                    dx, dy = 0, 0
                    if len(body) >= 2:
                        dx = body[0][0] - body[1][0]
                        dy = body[0][1] - body[1][1]
                    elif si < len(prev_heads) and prev_heads[si]:
                        ph = prev_heads[si]
                        dx = sx - ph[0]
                        dy = sy - ph[1]
                    # 网格环绕时归一化到 -1,0,1
                    if dx > 1: dx = -1
                    elif dx < -1: dx = 1
                    if dy > 1: dy = -1
                    elif dy < -1: dy = 1
                    if dx == 0 and dy == 0:
                        dx = 1
                    _draw_snake_head(screen, sx, sy, col, "triangle", dx, dy)

    return screen


def _pool_join_timeout(pool, timeout: float = 3.0) -> None:
    """等待 Pool 子进程退出，每个进程最多等待 timeout 秒，避免 join() 无限阻塞"""
    import time
    if not hasattr(pool, "_pool"):
        return
    t0 = time.perf_counter()
    for p in pool._pool:
        remain = max(0, timeout - (time.perf_counter() - t0))
        if remain > 0 and p.is_alive():
            p.join(timeout=remain)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)


def _init_pygame_worker():
    """子进程内初始化 pygame（仅渲染，无需显示）"""
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((1, 1))


def _process_one_item(item: tuple) -> dict:
    """处理单个样本：渲染 + 保存（在子进程中调用）"""
    scene, prev_scene, sc_idx, total_scenes, img_path, lbl_path, beh_path, metadata = item
    surf = render_scene(scene, scene_idx=sc_idx, prev_scene=prev_scene, total_scenes=total_scenes)
    pygame.image.save(surf, img_path)

    bboxes = scene_to_bboxes(scene)
    lbl_lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in bboxes]
    Path(lbl_path).write_text("\n".join(lbl_lines), encoding="utf-8")

    snake_ann = scene.get("snake_annotations", [])
    beh_data = {"snake_annotations": snake_ann}
    Path(beh_path).write_text(json.dumps(beh_data, ensure_ascii=False), encoding="utf-8")

    return metadata


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
    import multiprocessing as mp
    import random
    p = argparse.ArgumentParser(description="渲染 batch JSON 为 YOLO 数据集")
    p.add_argument("--batches", "-b", default=None, help="batch 目录，默认 batches/")
    p.add_argument("--output", "-o", default="dataset", help="输出根目录")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    p.add_argument("--workers", "-w", type=int, default=None, help="并行进程数，默认 CPU 核心数")
    args = p.parse_args()

    batches_dir = Path(args.batches or ROOT / "batches")
    output_dir = Path(args.output or ROOT / "dataset")
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

    all_samples: list[tuple[dict, dict | None, int, int, int, int, str]] = []
    for bf in batch_files:
        data = json.loads(bf.read_text(encoding="utf-8"))
        for ep_idx, ep in enumerate(data.get("episodes", [])):
            scenes = ep.get("scenes", [])
            for sc_idx, scene in enumerate(scenes):
                if not scene.get("snakes"):
                    continue
                prev = scenes[sc_idx - 1] if sc_idx > 0 else None
                all_samples.append((scene, prev, sc_idx, len(scenes), ep_idx, sc_idx, bf.name))

    n = len(all_samples)
    workers = min(args.workers or (mp.cpu_count() or 4), n)  # 不超过任务数
    val_n = int(n * args.val_ratio)
    train_n = n - val_n
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_idx = set(indices[:train_n])
    val_idx = set(indices[train_n:])

    # 构建任务列表：(scene, prev_scene, scene_idx, total_scenes, img_path, lbl_path, beh_path, metadata)
    work_items: list[tuple] = []
    for global_idx in range(n):
        scene, prev_scene, sc_idx, total_scenes, ep_idx, _, batch_name = all_samples[global_idx]
        split = "val" if global_idx in val_idx else "train"
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"
        beh_dir = output_dir / split / "behavior"
        name = f"{global_idx:06d}"
        metadata = {"id": name, "split": split, "batch": batch_name, "episode": ep_idx, "scene": sc_idx}
        work_items.append((
            scene, prev_scene, sc_idx, total_scenes,
            str(img_dir / f"{name}.png"),
            str(lbl_dir / f"{name}.txt"),
            str(beh_dir / f"{name}.json"),
            metadata,
        ))

    # 按 indices 顺序处理，确保 train/val 划分正确；work_items 已按 global_idx 排序
    metadata_list: list[dict] = []
    chunksize = max(1, n // (workers * 4))  # 适度分块减少 IPC

    if workers <= 1:
        # 单进程：主进程直接渲染，避免多进程开销
        pygame.init()
        pygame.display.set_mode((1, 1))
        for idx, item in enumerate(work_items):
            metadata_list.append(_process_one_item(item))
            if (idx + 1) % 500 == 0:
                print(f"已处理 {idx + 1}/{n} ...")
    else:
        print(f"使用 {workers} 个进程并行渲染...")
        pool = mp.Pool(processes=workers, initializer=_init_pygame_worker)
        interrupted = False
        try:
            for idx, meta in enumerate(pool.imap(_process_one_item, work_items, chunksize=chunksize)):
                metadata_list.append(meta)
                if (idx + 1) % 500 == 0:
                    print(f"已处理 {idx + 1}/{n} ...")
        except KeyboardInterrupt:
            interrupted = True
            print("\n[中断] 用户取消，正在终止工作进程...", flush=True)
            pool.terminate()
            _pool_join_timeout(pool, timeout=3.0)
            raise SystemExit(130)
        finally:
            if not interrupted:
                pool.close()
                pool.join()

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

    print(f"完成: train {train_n}, val {val_n}, 总计 {n} (进程数 {workers})")
    print(f"输出: {output_dir}")
    print(f"配置: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

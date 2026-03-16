"""
训练行为正确性识别模型

支持两种数据源:
  1. track_sequences.json (YOLO 跟踪输出): --data sequences/track_sequences.json
  2. 纯网格 (batch JSON，无需 YOLO): --data grid --batches batches/

增强选项（缓解类别不平衡、提升泛化）:
  --class-weights    类别权重（逆频率）
  --oversample       过采样少数类
  --aug-multiscale   多尺度时间（随机 1/2/3 倍采样子序列）
  --aug-frame-drop   随机丢帧概率
  --aug-noise        特征高斯噪声 std

用法:
  python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior
  python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior \\
    --class-weights --oversample --aug-multiscale --aug-frame-drop 0.1 --aug-noise 0.02
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GRID_W = 15
GRID_H = 15


def _norm(x: float, size: int) -> float:
    """网格坐标归一化到 [0,1]"""
    return (x + 0.5) / size


def _head_forward_type_from_scene(scene: dict) -> list[int]:
    """蛇头前方一格: 0=空, 1=自己身体, 2=其他蛇身体, 3=其他蛇头"""
    snakes_data = scene.get("snakes", [])
    if not snakes_data:
        return []
    result = []
    for si, s in enumerate(snakes_data):
        body = s.get("body", [])
        if not body:
            result.append(0)
            continue
        try:
            hx, hy = int(body[0][0]), int(body[0][1])
        except (IndexError, TypeError):
            result.append(0)
            continue
        dx, dy = (1, 0)
        if len(body) >= 2:
            dx = int(body[0][0]) - int(body[1][0])
            dy = int(body[0][1]) - int(body[1][1])
        fx = ((hx + dx) % GRID_W + GRID_W) % GRID_W
        fy = ((hy + dy) % GRID_H + GRID_H) % GRID_H
        own_cells = {((int(p[0]) % GRID_W + GRID_W) % GRID_W, (int(p[1]) % GRID_H + GRID_H) % GRID_H) for p in body}
        other_heads = set()
        other_bodies = set()
        for sj, s2 in enumerate(snakes_data):
            if sj == si:
                continue
            for idx, p in enumerate(s2.get("body", [])):
                try:
                    gx = (int(p[0]) % GRID_W + GRID_W) % GRID_W
                    gy = (int(p[1]) % GRID_H + GRID_H) % GRID_H
                    if idx == 0:
                        other_heads.add((gx, gy))
                    else:
                        other_bodies.add((gx, gy))
                except (IndexError, TypeError):
                    pass
        fwd = (fx, fy)
        if fwd in own_cells:
            result.append(1)
        elif fwd in other_heads:
            result.append(3)
        elif fwd in other_bodies:
            result.append(2)
        else:
            result.append(0)
    return result


def load_track_sequences(
    path: Path, add_velocity: bool = True
) -> tuple[list[tuple[list[list[float]], int, int, int]], list[tuple[str, int]], list[str] | None]:
    """
    加载 track_sequences.json，返回 (samples, episode_keys, splits)。
    samples: [(features, label_idx, reason_idx, is_endpoint), ...]
    episode_keys: 每条样本对应的 (batch, episode)
    splits: 每条样本的 "train" 或 "val"，若文件中无 split 则为 None（调用方按 episode 随机划分）
    """
    from models.behavior_correctness import REASON_TO_IDX

    data = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    episode_keys = []
    splits: list[str] = []
    for rec in data:
        feats = rec.get("features", [])
        if len(feats) < 2:
            continue
        feats = sorted(feats, key=lambda x: x["t"])
        seq_cont, seq_hf = [], []
        for i, f in enumerate(feats):
            xc, yc = f["xc"], f["yc"]
            fx = f.get("fx", 0.0)
            fy = f.get("fy", 0.0)
            xx = f.get("xx", 0.0)
            xy = f.get("xy", 0.0)
            has_x2 = f.get("has_x2", 0.0)
            ate_food = f.get("ate_food", 0.0)
            ate_x2 = f.get("ate_x2", 0.0)
            ate_food_while_x2 = f.get("ate_food_while_x2_exists", 1.0 if (ate_food and has_x2) else 0.0)
            head_forward = max(0, min(3, int(f.get("head_forward_type", 0))))
            is_dead = f.get("is_dead", 0.0)
            steps_since_food = f.get("steps_since_food", 0.0)
            about_to_timeout = f.get("about_to_timeout", (1.0 if steps_since_food >= 79.0 / 80.0 else 0.0))
            if add_velocity and i > 0:
                dx = xc - feats[i - 1]["xc"]
                dy = yc - feats[i - 1]["yc"]
            else:
                dx, dy = 0.0, 0.0
            df = ((fx - xc) ** 2 + (fy - yc) ** 2) ** 0.5 if (fx or fy) else 0.0
            dx2 = ((xx - xc) ** 2 + (xy - yc) ** 2) ** 0.5 if has_x2 and (xx or xy) else 0.0
            df = min(df, 1.5)
            dx2 = min(dx2, 1.5)
            vel_norm = (dx * dx + dy * dy) ** 0.5 or 1e-6
            to_food = (fx - xc) * dx + (fy - yc) * dy
            move_to_food = max(-1, min(1, to_food / vel_norm)) if (fx or fy) else 0.0
            cont = [xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food, ate_food, ate_x2, is_dead, steps_since_food, ate_food_while_x2, about_to_timeout]
            seq_cont.append(cont)
            seq_hf.append(head_forward)
        if not add_velocity:
            seq_cont, seq_hf = [], []
            for f in feats:
                xc, yc = f["xc"], f["yc"]
                fx, fy = f.get("fx",0), f.get("fy",0)
                xx, xy = f.get("xx",0), f.get("xy",0)
                hx2 = f.get("has_x2",0)
                ate_food, ate_x2 = f.get("ate_food",0), f.get("ate_x2",0)
                ate_food_while_x2 = f.get("ate_food_while_x2_exists", 1.0 if (ate_food and hx2) else 0.0)
                head_forward = max(0, min(3, int(f.get("head_forward_type", 0))))
                is_dead = f.get("is_dead", 0.0)
                steps_since_food = f.get("steps_since_food", 0.0)
                about_to_timeout = f.get("about_to_timeout", (1.0 if steps_since_food >= 79.0 / 80.0 else 0.0))
                df = min(((fx-xc)**2+(fy-yc)**2)**0.5, 1.5) if (fx or fy) else 0.0
                dx2 = min(((xx-xc)**2+(xy-yc)**2)**0.5, 1.5) if hx2 and (xx or xy) else 0.0
                seq_cont.append([xc, yc, 0, 0, fx, fy, xx, xy, hx2, df, dx2, 0.0, ate_food, ate_x2, is_dead, steps_since_food, ate_food_while_x2, about_to_timeout])
                seq_hf.append(head_forward)

        label = rec.get("label", "correct")
        label_idx = 1 if label == "incorrect" else 0
        reason = rec.get("reason", "in_progress")
        reason_idx = REASON_TO_IDX.get(reason, REASON_TO_IDX["in_progress"])
        is_endpoint = int(rec.get("is_endpoint", 1))
        samples.append(((seq_cont, seq_hf), label_idx, reason_idx, is_endpoint))
        episode_keys.append((rec.get("batch", ""), rec.get("episode", 0)))
        splits.append(rec.get("split", ""))
    # 若所有记录都有有效 split，返回 splits；否则返回 None 供调用方按 episode 随机划分
    use_splits = splits if all(s in ("train", "val") for s in splits) else None
    return samples, episode_keys, use_splits


def load_grid_sequences(
    batches_dir: Path, add_velocity: bool = True
) -> tuple[list[tuple[list[list[float]], int, int, int]], list[tuple[str, int]]]:
    """
    从 batch JSON 加载网格序列，特征必须与 YOLO/label 路径一致（仅用 YOLO 可检测的 head/food/x2）。
    不使用 x2_active、score 等游戏内部状态。
    输出 18 维与 track 一致: 上述 17 维 + about_to_timeout（下一步再不吃就超时为 1）
    """
    from models.behavior_correctness import REASON_TO_IDX

    batch_files = sorted(batches_dir.glob("batch_*.json"))
    samples = []
    episode_keys = []
    for bf in batch_files:
        data = json.loads(bf.read_text(encoding="utf-8"))
        for ep_idx, ep in enumerate(data.get("episodes", [])):
            scenes = ep.get("scenes", [])
            if not scenes:
                continue
            snakes_data = scenes[0].get("snakes", [])
            num_snakes = len(snakes_data)
            for si in range(num_snakes):
                raw_frames: list[tuple[float, float, float, float, float, float, float, dict]] = []
                for sc in scenes:
                    snakes = sc.get("snakes", [])
                    if si >= len(snakes):
                        break
                    s = snakes[si]
                    body = s.get("body", [])
                    food = s.get("food", [0, 0])
                    x2 = s.get("x2")
                    if not body:
                        continue
                    hx, hy = body[0][0], body[0][1]
                    fx, fy = int(food[0]) if food else 0, int(food[1]) if food else 0
                    xx, xy = (int(x2[0]), int(x2[1])) if x2 else (0, 0)
                    xc = _norm(hx, GRID_W)
                    yc = _norm(hy, GRID_H)
                    raw_frames.append((xc, yc, _norm(fx, GRID_W), _norm(fy, GRID_H),
                                       _norm(xx, GRID_W) if x2 else 0, _norm(xy, GRID_H) if x2 else 0,
                                       1.0 if x2 else 0.0, sc))
                if len(raw_frames) < 2:
                    continue
                anns = scenes[-1].get("snake_annotations", [])
                is_dead_last = 0.0
                if si < len(anns):
                    r = anns[si].get("reason", "in_progress")
                    if r in ("self_collision", "snake_collision"):
                        is_dead_last = 1.0
                seq_cont, seq_hf = [], []
                steps_counter = 0
                for i, (xc, yc, fx, fy, xx, xy, has_x2, sc) in enumerate(raw_frames):
                    hf_list = _head_forward_type_from_scene(sc)
                    head_forward = int(hf_list[si]) if si < len(hf_list) else 0
                    head_forward = max(0, min(3, head_forward))
                    p = raw_frames[i - 1][:7] if i > 0 else None
                    ate_food, ate_x2 = 0.0, 0.0
                    if p:
                        thresh = 0.02
                        def _d(ax, ay, bx, by): return (ax - bx) ** 2 + (ay - by) ** 2
                        if (p[2] or p[3]) and _d(fx, fy, p[2], p[3]) > thresh and _d(xc, yc, p[2], p[3]) < thresh:
                            ate_food = 1.0
                            steps_counter = 0
                        if p[6] and (not has_x2 or _d(xx, xy, p[4], p[5]) > thresh) and (p[4] or p[5]) and _d(xc, yc, p[4], p[5]) < thresh:
                            ate_x2 = 1.0
                    ate_food_while_x2 = 1.0 if (ate_food and has_x2) else 0.0
                    if ate_food == 0:
                        steps_counter += 1
                    steps_since_food = min(steps_counter / 80.0, 1.0)
                    about_to_timeout = 1.0 if steps_counter >= 79 else 0.0
                    is_dead = is_dead_last if i == len(raw_frames) - 1 else 0.0
                    if add_velocity and i > 0 and p is not None:
                        dx, dy = xc - p[0], yc - p[1]
                        df = min(((fx - xc) ** 2 + (fy - yc) ** 2) ** 0.5, 1.5) if (fx or fy) else 0.0
                        dx2 = min(((xx - xc) ** 2 + (xy - yc) ** 2) ** 0.5, 1.5) if has_x2 and (xx or xy) else 0.0
                        vel = (dx * dx + dy * dy) ** 0.5 or 1e-6
                        to_food = (fx - xc) * dx + (fy - yc) * dy
                        move_to_food = max(-1, min(1, to_food / vel)) if (fx or fy) else 0.0
                    else:
                        dx, dy = 0.0, 0.0
                        df = min(((fx - xc) ** 2 + (fy - yc) ** 2) ** 0.5, 1.5) if (fx or fy) else 0.0
                        dx2 = min(((xx - xc) ** 2 + (xy - yc) ** 2) ** 0.5, 1.5) if has_x2 and (xx or xy) else 0.0
                        move_to_food = 0.0
                    cont = [xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food, ate_food, ate_x2, is_dead, steps_since_food, ate_food_while_x2, about_to_timeout]
                    seq_cont.append(cont)
                    seq_hf.append(head_forward)
                if len(seq_cont) < 2:
                    continue
                anns = scenes[-1].get("snake_annotations", [])
                if si >= len(anns):
                    continue
                ann = anns[si]
                label_idx = 1 if ann.get("label") == "incorrect" else 0
                reason = ann.get("reason", "in_progress")
                reason_idx = REASON_TO_IDX.get(reason, REASON_TO_IDX["in_progress"])
                samples.append(((seq_cont, seq_hf), label_idx, reason_idx, 1))
                episode_keys.append((bf.name, ep_idx))
    return samples, episode_keys


def main():
    import argparse
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from models.behavior_correctness import BehaviorCorrectnessModel, NUM_REASONS

    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", required=True,
                   help="track_sequences.json 路径，或 grid 表示从 batch 加载")
    p.add_argument("--batches", "-b", default="batches",
                   help="batch 目录（--data grid 时使用）")
    p.add_argument("--output", "-o", default="checkpoints/behavior",
                   help="输出目录")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--hidden", type=int, default=128, help="LSTM 隐藏维")
    p.add_argument("--lstm-layers", type=int, default=2, help="LSTM 层数")
    p.add_argument("--dropout", type=float, default=0.3, help="LSTM dropout")
    p.add_argument("--no-bidirectional", action="store_true", help="禁用双向 LSTM")
    p.add_argument("--no-attention", action="store_true", help="禁用自注意力")
    p.add_argument("--label-smoothing", type=float, default=0.1, help="CE label smoothing")
    p.add_argument("--input-dim", type=int, default=0,
                   help="0=自动: 16*frame_context，默认 16*3=48")
    p.add_argument("--frame-context", type=int, default=3,
                   help="前后帧合并数：1 前 + 当前 + 1 后 = 3。1 表示不合并")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-velocity", action="store_true",
                   help="track 数据不加速度特征")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="L2 正则，缓解过拟合")
    # 类别不平衡与增强
    p.add_argument("--class-weights", action="store_true",
                   help="使用类别权重（逆频率）缓解不平衡")
    p.add_argument("--incorrect-weight", type=float, default=0.0,
                   help="incorrect 类额外权重倍数，如 2.0 表示 incorrect 权重 x2（用于提升错误检测率）")
    p.add_argument("--focal-loss", action="store_true",
                   help="对 correct/incorrect 使用 Focal Loss，聚焦难分样本")
    p.add_argument("--focal-gamma", type=float, default=2.0,
                   help="Focal Loss 的 gamma 参数，越大越关注难样本")
    p.add_argument("--oversample", action="store_true",
                   help="过采样少数类，使用 WeightedRandomSampler")
    p.add_argument("--boost-incorrect", action="store_true",
                   help="一键启用: class-weights + oversample + focal-loss + incorrect-weight=2，提升错误检测率")
    p.add_argument("--aug-frame-drop", type=float, default=0.0,
                   help="训练时随机丢帧概率 (0=关闭)")
    p.add_argument("--aug-noise", type=float, default=0.0,
                   help="特征高斯噪声 std (0=关闭)")
    p.add_argument("--aug-multiscale", action="store_true",
                   help="多尺度时间：随机取 1/2/3 倍采样子序列")
    p.add_argument("--patience", type=int, default=50,
                   help="早停：验证指标连续 N 个 epoch 无提升则停止，0=禁用")
    args = p.parse_args()
    if args.boost_incorrect:
        args.class_weights = True
        args.oversample = True
        args.focal_loss = True
        args.incorrect_weight = args.incorrect_weight or 2.0
        print("已启用 --boost-incorrect: class-weights + oversample + focal-loss + incorrect-weight=2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame_ctx = max(1, args.frame_context)
    if frame_ctx % 2 == 0:
        frame_ctx += 1  # 确保奇数：half 前 + 当前 + half 后
    half = frame_ctx // 2

    if args.data.lower() == "grid":
        batches_dir = Path(args.batches)
        if not batches_dir.is_absolute():
            batches_dir = ROOT / batches_dir
        samples, episode_keys = load_grid_sequences(batches_dir, add_velocity=not args.no_velocity)
        splits = None  # grid 无 split，后续按 episode 随机划分
        base_cont_dim = 18
        input_dim = args.input_dim or (base_cont_dim * frame_ctx if frame_ctx > 1 else base_cont_dim)
        print(f"从 grid 加载: {len(samples)} 条序列 (与 YOLO 路径同特征), base_cont_dim={base_cont_dim}, frame_ctx={frame_ctx}, input_dim={input_dim}")
    else:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        if not data_path.exists():
            print(f"文件不存在: {data_path}")
            print("请先运行: python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences")
            sys.exit(1)
        samples, episode_keys, splits = load_track_sequences(data_path, add_velocity=not args.no_velocity)
        base_cont_dim = len(samples[0][0][0][0]) if samples else 18
        input_dim = args.input_dim or (base_cont_dim * frame_ctx if frame_ctx > 1 else base_cont_dim)
        print(f"从 track_sequences 加载: {len(samples)} 条序列, base_cont_dim={base_cont_dim}, frame_ctx={frame_ctx}, input_dim={input_dim}")

    if not samples:
        print("无有效样本")
        sys.exit(1)

    # 优先使用数据中的 split（与 render_and_export / run_track_and_prepare 一致），否则按 episode 随机划分
    if splits is not None:
        train_samples = [s for s, sp in zip(samples, splits) if sp == "train"]
        val_samples = [s for s, sp in zip(samples, splits) if sp == "val"]
        print(f"使用数据中的 split 划分: train {len(train_samples)} 条, val {len(val_samples)} 条")
    else:
        unique_eps = list(dict.fromkeys(episode_keys))
        random.shuffle(unique_eps)
        val_n = int(len(unique_eps) * args.val_ratio)
        val_eps = set(unique_eps[:val_n])
        train_samples = [s for s, ek in zip(samples, episode_keys) if ek not in val_eps]
        val_samples = [s for s, ek in zip(samples, episode_keys) if ek in val_eps]
        print(f"按 episode 划分（数据无 split）: train {len(train_samples)} 条, val {len(val_samples)} 条 ({len(val_eps)} 个 val episode)")
    if not val_samples:
        print("警告: 验证集为空，请减小 --val-ratio 或检查数据 split")
        sys.exit(1)

    # 类别权重：逆频率，用于 reason (7 类)
    from models.behavior_correctness import NUM_REASONS
    reason_counts = np.bincount([s[2] for s in train_samples if s[3] > 0.5], minlength=NUM_REASONS)
    reason_counts = np.maximum(reason_counts, 1)
    reason_weights = 1.0 / np.sqrt(reason_counts)
    reason_weights = reason_weights / reason_weights.sum() * NUM_REASONS
    label_counts = np.bincount([s[1] for s in train_samples if s[3] > 0.5], minlength=2)
    label_counts = np.maximum(label_counts, 1)
    label_weights = 1.0 / np.sqrt(label_counts)
    label_weights = label_weights / label_weights.sum() * 2
    if args.incorrect_weight > 0:
        label_weights[1] *= args.incorrect_weight  # index 1 = incorrect
        label_weights = label_weights / label_weights.sum() * 2
        print(f"incorrect 额外权重: x{args.incorrect_weight} -> label_weights={label_weights.tolist()}")
    if args.class_weights:
        print(f"类别权重: label={label_weights.tolist()}, reason(前3)={reason_weights[:3].tolist()}...")
    if args.focal_loss:
        print(f"Focal Loss: gamma={args.focal_gamma} (聚焦难分样本)")

    # 过采样：按 reason 逆频率赋权，incorrect 额外加权
    def _sample_weights(data):
        reason_ct = np.bincount([s[2] for s in data if s[3] > 0.5], minlength=NUM_REASONS)
        reason_ct = np.maximum(reason_ct, 1)
        inv = 1.0 / np.sqrt(reason_ct)
        weights = [inv[s[2]] if s[3] > 0.5 else inv.mean() for s in data]
        if args.incorrect_weight > 0:
            weights = [w * (args.incorrect_weight if s[1] == 1 else 1.0) for w, s in zip(weights, data)]
        return weights

    def _merge_frame_context_cont(seq: list, base_dim: int, half: int) -> list:
        """将前后各 half 帧并入当前帧。seq 每项 base_dim 维，输出每项 base_dim*(2*half+1) 维。"""
        if half <= 0:
            return seq
        n = len(seq)
        out = []
        for i in range(n):
            ctx = []
            for j in range(i - half, i + half + 1):
                idx = max(0, min(n - 1, j))
                ctx.extend(seq[idx])
            out.append(ctx)
        return out

    class SeqDataset(Dataset):
        def __init__(self, data, augment=False, frame_context_half=0, base_cont_dim=18):
            self.data = data
            self.augment = augment
            self.frame_half = frame_context_half
            self.base_cont_dim = base_cont_dim

        def __len__(self):
            return len(self.data)

        def _augment_indices(self, n: int) -> np.ndarray:
            """返回保留的索引"""
            if n < 3:
                return np.arange(n)
            idx = np.arange(n)
            if self.augment and args.aug_multiscale and np.random.rand() < 0.5:
                step = np.random.choice([1, 2, 3])
                idx = np.arange(0, n, step)
                if idx[-1] != n - 1:
                    idx = np.concatenate([idx, [n - 1]])
            if self.augment and args.aug_frame_drop > 0:
                keep = np.ones(len(idx), dtype=bool)
                keep[0] = keep[-1] = True
                for i in range(1, len(idx) - 1):
                    if np.random.rand() < args.aug_frame_drop:
                        keep[i] = False
                idx = idx[keep]
            return idx

        def __getitem__(self, i):
            (seq_cont, seq_hf), label, reason, is_ep = self.data[i]
            seq_cont = [list(x) for x in seq_cont]
            n = len(seq_cont)
            idx = self._augment_indices(n)
            seq_cont = [seq_cont[j] for j in idx]
            seq_hf = [seq_hf[j] for j in idx]
            if self.augment and args.aug_noise > 0:
                arr = np.array(seq_cont, dtype=np.float32)
                arr = arr + np.random.randn(*arr.shape).astype(np.float32) * args.aug_noise
                seq_cont = arr.tolist()
            if self.frame_half > 0:
                seq_cont = _merge_frame_context_cont(seq_cont, self.base_cont_dim, self.frame_half)
            return torch.tensor(seq_cont, dtype=torch.float32), torch.tensor(seq_hf, dtype=torch.long), label, reason, is_ep

    def collate(batch):
        conts, hfs, labels, reasons, is_eps = zip(*batch)
        lengths = torch.tensor([c.size(0) for c in conts])
        padded_cont = nn.utils.rnn.pad_sequence(conts, batch_first=True, padding_value=0)
        padded_hf = nn.utils.rnn.pad_sequence(hfs, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        reasons = torch.tensor(reasons, dtype=torch.long)
        is_endpoints = torch.tensor(is_eps, dtype=torch.float32).unsqueeze(1)
        return padded_cont, padded_hf, lengths, labels, reasons, is_endpoints

    train_ds = SeqDataset(
        train_samples, augment=True,
        frame_context_half=half if frame_ctx > 1 else 0,
        base_cont_dim=base_cont_dim,
    )
    sampler = None
    if args.oversample:
        sw = _sample_weights(train_samples)
        sampler = torch.utils.data.WeightedRandomSampler(sw, len(sw))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        SeqDataset(
            val_samples, augment=False,
            frame_context_half=half if frame_ctx > 1 else 0,
            base_cont_dim=base_cont_dim,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCorrectnessModel(
        input_dim=input_dim,
        hidden_dim=args.hidden,
        num_layers=args.lstm_layers,
        dropout=args.dropout,
        bidirectional=not args.no_bidirectional,
        use_attention=not args.no_attention,
        use_head_forward_embedding=True,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=25)

    try:
        from sklearn.metrics import f1_score
    except ImportError:
        print("请安装 sklearn: pip install scikit-learn")
        sys.exit(1)

    def _focal_loss(logits, targets, gamma, weight=None):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** gamma) * ce
        if weight is not None:
            alpha_t = weight[targets]
            focal = alpha_t * focal
        return focal.mean()

    ce_kw = {"label_smoothing": args.label_smoothing}
    if args.class_weights or args.incorrect_weight > 0:
        ce_kw["weight"] = torch.tensor(label_weights, dtype=torch.float32).to(device)
    ce_label = nn.CrossEntropyLoss(**ce_kw)
    ce_kw_reason = {"label_smoothing": args.label_smoothing}
    if args.class_weights:
        ce_kw_reason["weight"] = torch.tensor(reason_weights, dtype=torch.float32).to(device)
    ce_reason = nn.CrossEntropyLoss(**ce_kw_reason)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("最优模型选取: mAP  |  早停 patience={}".format(args.patience))

    best_val_f1 = 0.0
    best_epoch = 0
    epochs_without_improve = 0
    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, x_hf, lengths, y_label, y_reason, y_ep in train_loader:
            x, x_hf = x.to(device), x_hf.to(device)
            lengths = lengths.to(device)
            y_label, y_reason = y_label.to(device), y_reason.to(device)
            y_ep = y_ep.to(device)
            opt.zero_grad()
            logits_c, logits_r = model(x, lengths, x_head_forward=x_hf)
            mask = (y_ep > 0.5).squeeze(1)
            if mask.any():
                lc = logits_c[mask]
                yl = y_label[mask]
                if args.focal_loss:
                    loss_c = _focal_loss(
                        lc, yl, args.focal_gamma,
                        ce_kw.get("weight"),
                    )
                else:
                    loss_c = ce_label(lc, yl)
                loss_r = ce_reason(logits_r[mask], y_reason[mask])
                loss = 0.5 * (loss_c + loss_r)
                train_correct += (logits_c[mask].argmax(1) == y_label[mask]).sum().item()
                train_total += mask.sum().item()
                loss.backward()
                opt.step()
                train_loss += loss.item()

        model.eval()
        val_preds, val_labels = [], []
        val_reason_preds, val_reason_labels = [], []
        val_reason_probs_list = []
        val_prob_incorrect_list = []
        with torch.no_grad():
            for x, x_hf, lengths, y_label, y_reason, y_ep in val_loader:
                x, x_hf = x.to(device), x_hf.to(device)
                lengths = lengths.to(device)
                y_label, y_reason, y_ep = y_label.to(device), y_reason.to(device), y_ep.to(device)
                logits_c, logits_r = model(x, lengths, x_head_forward=x_hf)
                mask = (y_ep > 0.5).squeeze(1)
                if mask.any():
                    pred = logits_c[mask].argmax(1).cpu().numpy()
                    gt = y_label[mask].cpu().numpy()
                    val_preds.extend(pred.tolist())
                    val_labels.extend(gt.tolist())
                    r_pred = logits_r[mask].argmax(1).cpu().numpy()
                    r_gt = y_reason[mask].cpu().numpy()
                    val_reason_preds.extend(r_pred.tolist())
                    val_reason_labels.extend(r_gt.tolist())
                    prob_r = torch.softmax(logits_r[mask], dim=1).cpu().numpy()
                    val_reason_probs_list.append(prob_r)
                    prob_incorrect = torch.softmax(logits_c[mask], dim=1).cpu().numpy()[:, 1]
                    val_prob_incorrect_list.append(prob_incorrect)

        binary_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0) if val_preds else 0.0
        reason_f1 = f1_score(val_reason_labels, val_reason_preds, average="macro", zero_division=0) if val_reason_preds else 0.0
        # 与 eval_behavior 一致：在固定阈值 0.5 下构造 effective reason 再算 mAP，便于与评估脚本对比
        from sklearn.metrics import average_precision_score
        from sklearn.preprocessing import label_binarize
        from models.behavior_correctness import NUM_REASONS
        if val_reason_probs_list:
            val_reason_probs = np.concatenate(val_reason_probs_list, axis=0)
            val_prob_incorrect = np.concatenate(val_prob_incorrect_list, axis=0)
            val_reason_labels_arr = np.array(val_reason_labels)
            y_bin = label_binarize(val_reason_labels_arr, classes=range(NUM_REASONS))
            pred_b = (val_prob_incorrect >= 0.5).astype(np.int64)
            effective_probs = np.zeros_like(val_reason_probs)
            for i in range(len(pred_b)):
                if pred_b[i] == 0:
                    effective_probs[i, 2] = 1.0
                else:
                    effective_probs[i, :] = val_reason_probs[i, :]
            ap_per_class = []
            for c in range(NUM_REASONS):
                if y_bin[:, c].sum() == 0:
                    ap_per_class.append(0.0)
                else:
                    ap = average_precision_score(y_bin[:, c], effective_probs[:, c])
                    ap_per_class.append(ap)
            select_score = float(np.mean(ap_per_class)) if ap_per_class else 0.0
        else:
            select_score = 0.0

        scheduler.step(select_score)
        if select_score > best_val_f1:
            best_val_f1 = select_score
            best_epoch = ep + 1
            epochs_without_improve = 0
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": args.hidden,
                "num_layers": args.lstm_layers,
                "bidirectional": not args.no_bidirectional,
                "use_attention": not args.no_attention,
                "use_head_forward_embedding": True,
            }, out_dir / "best.pt")
        else:
            epochs_without_improve += 1
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden,
            "num_layers": args.lstm_layers,
            "bidirectional": not args.no_bidirectional,
            "use_attention": not args.no_attention,
            "use_head_forward_embedding": True,
        }, out_dir / "last.pt")
        print(f"Epoch {ep+1}/{args.epochs} loss={train_loss/len(train_loader):.4f} mAP={select_score:.4f}")

        if args.patience > 0 and epochs_without_improve >= args.patience:
            print(f"早停: {args.patience} epoch 无提升，停止于 epoch {ep + 1}")
            break

    print(f"训练完成，最佳 mAP={best_val_f1:.4f} (epoch {best_epoch})")
    print(f"模型保存: {out_dir / 'best.pt'}, {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()

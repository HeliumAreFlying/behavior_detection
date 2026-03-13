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


def load_track_sequences(
    path: Path, add_velocity: bool = True
) -> tuple[list[tuple[list[list[float]], int, int, int]], list[tuple[str, int]]]:
    """
    加载 track_sequences.json，返回 (samples, episode_keys)。
    samples: [(features, label_idx, reason_idx, is_endpoint), ...]
    episode_keys: 每条样本对应的 (batch, episode)，用于按 episode 划分 train/val
    """
    from models.behavior_correctness import REASON_TO_IDX

    data = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    episode_keys = []
    for rec in data:
        feats = rec.get("features", [])
        if len(feats) < 2:
            continue
        feats = sorted(feats, key=lambda x: x["t"])
        seq = []
        for i, f in enumerate(feats):
            xc, yc = f["xc"], f["yc"]
            fx = f.get("fx", 0.0)
            fy = f.get("fy", 0.0)
            xx = f.get("xx", 0.0)
            xy = f.get("xy", 0.0)
            has_x2 = f.get("has_x2", 0.0)
            if add_velocity and i > 0:
                dx = xc - feats[i - 1]["xc"]
                dy = yc - feats[i - 1]["yc"]
            else:
                dx, dy = 0.0, 0.0
            # 距离与方向：dist_to_food, dist_to_x2, moving_towards_food
            df = ((fx - xc) ** 2 + (fy - yc) ** 2) ** 0.5 if (fx or fy) else 0.0
            dx2 = ((xx - xc) ** 2 + (xy - yc) ** 2) ** 0.5 if has_x2 and (xx or xy) else 0.0
            df = min(df, 1.5)
            dx2 = min(dx2, 1.5)
            vel_norm = (dx * dx + dy * dy) ** 0.5 or 1e-6
            to_food = (fx - xc) * dx + (fy - yc) * dy
            move_to_food = max(-1, min(1, to_food / vel_norm)) if (fx or fy) else 0.0
            # 12 dims: head, vel, food, x2, has_x2, dist_food, dist_x2, move_to_food
            seq.append([xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food])
        if not add_velocity:
            seq = []
            for f in feats:
                xc, yc = f["xc"], f["yc"]
                fx, fy = f.get("fx",0), f.get("fy",0)
                xx, xy = f.get("xx",0), f.get("xy",0)
                hx2 = f.get("has_x2",0)
                df = min(((fx-xc)**2+(fy-yc)**2)**0.5, 1.5) if (fx or fy) else 0.0
                dx2 = min(((xx-xc)**2+(xy-yc)**2)**0.5, 1.5) if hx2 and (xx or xy) else 0.0
                seq.append([xc, yc, 0, 0, fx, fy, xx, xy, hx2, df, dx2, 0.0])

        label = rec.get("label", "correct")
        label_idx = 1 if label == "incorrect" else 0
        reason = rec.get("reason", "in_progress")
        reason_idx = REASON_TO_IDX.get(reason, REASON_TO_IDX["in_progress"])
        is_endpoint = int(rec.get("is_endpoint", 1))
        samples.append((seq, label_idx, reason_idx, is_endpoint))
        episode_keys.append((rec.get("batch", ""), rec.get("episode", 0)))
    return samples, episode_keys


def load_grid_sequences(
    batches_dir: Path
) -> tuple[list[tuple[list[list[float]], int, int, int]], list[tuple[str, int]]]:
    """
    从 batch JSON 加载网格序列: 每蛇每食物周期为一条样本
    features: [head_x_norm, head_y_norm, food_x_norm, food_y_norm, x2_x_norm, x2_y_norm, has_x2, score_norm]
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
                seq = []
                for sc in scenes:
                    snakes = sc.get("snakes", [])
                    if si >= len(snakes):
                        break
                    s = snakes[si]
                    body = s.get("body", [])
                    food = s.get("food", [0, 0])
                    x2 = s.get("x2")
                    score = s.get("score", 0)
                    x2_active = s.get("x2_active", False)
                    if not body:
                        continue
                    hx, hy = body[0][0], body[0][1]
                    fx, fy = int(food[0]) if food else 0, int(food[1]) if food else 0
                    xx, xy = (int(x2[0]), int(x2[1])) if x2 else (0, 0)
                    feat = [
                        _norm(hx, GRID_W), _norm(hy, GRID_H),
                        _norm(fx, GRID_W), _norm(fy, GRID_H),
                        _norm(xx, GRID_W) if x2 else 0, _norm(xy, GRID_H) if x2 else 0,
                        1.0 if x2_active else 0.0,
                        min(score / 24.0, 1.0),
                    ]
                    seq.append(feat)
                if len(seq) < 2:
                    continue
                anns = scenes[-1].get("snake_annotations", [])
                if si >= len(anns):
                    continue
                ann = anns[si]
                label_idx = 1 if ann.get("label") == "incorrect" else 0
                reason = ann.get("reason", "in_progress")
                reason_idx = REASON_TO_IDX.get(reason, REASON_TO_IDX["in_progress"])
                samples.append((seq, label_idx, reason_idx, 1))
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
    p.add_argument("--label-smoothing", type=float, default=0.1, help="CE label smoothing")
    p.add_argument("--input-dim", type=int, default=0,
                   help="0=自动: track 用 4，grid 用 8")
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

    if args.data.lower() == "grid":
        batches_dir = Path(args.batches)
        if not batches_dir.is_absolute():
            batches_dir = ROOT / batches_dir
        samples, episode_keys = load_grid_sequences(batches_dir)
        input_dim = args.input_dim or 8
        print(f"从 grid 加载: {len(samples)} 条序列, input_dim={input_dim}")
    else:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        if not data_path.exists():
            print(f"文件不存在: {data_path}")
            print("请先运行: python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences")
            sys.exit(1)
        samples, episode_keys = load_track_sequences(data_path, add_velocity=not args.no_velocity)
        input_dim = args.input_dim or (len(samples[0][0][0]) if samples else 9)
        print(f"从 track_sequences 加载: {len(samples)} 条序列, input_dim={input_dim}")

    if not samples:
        print("无有效样本")
        sys.exit(1)

    # 按 episode 划分，避免同一局样本泄漏到 train/val
    unique_eps = list(dict.fromkeys(episode_keys))
    random.shuffle(unique_eps)
    val_n = int(len(unique_eps) * args.val_ratio)
    val_eps = set(unique_eps[:val_n])
    train_samples = [s for s, ek in zip(samples, episode_keys) if ek not in val_eps]
    val_samples = [s for s, ek in zip(samples, episode_keys) if ek in val_eps]
    print(f"按 episode 划分: train {len(train_samples)} 条, val {len(val_samples)} 条 ({len(val_eps)} 个 val episode)")
    if not val_samples:
        print("警告: 验证集为空，请减小 --val-ratio")
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

    class SeqDataset(Dataset):
        def __init__(self, data, augment=False):
            self.data = data
            self.augment = augment

        def __len__(self):
            return len(self.data)

        def _augment_seq(self, seq):
            """多尺度+丢帧+噪声"""
            arr = np.array(seq, dtype=np.float32)
            n = len(arr)
            if n < 3:
                return arr
            if self.augment and args.aug_multiscale and np.random.rand() < 0.5:
                step = np.random.choice([1, 2, 3])
                idx = np.arange(0, n, step)
                if idx[-1] != n - 1:
                    idx = np.concatenate([idx, [n - 1]])
                arr = arr[idx]
            if self.augment and args.aug_frame_drop > 0:
                keep = np.ones(len(arr), dtype=bool)
                keep[0] = keep[-1] = True
                for i in range(1, len(arr) - 1):
                    if np.random.rand() < args.aug_frame_drop:
                        keep[i] = False
                arr = arr[keep]
            if self.augment and args.aug_noise > 0:
                arr = arr + np.random.randn(*arr.shape).astype(np.float32) * args.aug_noise
            return arr

        def __getitem__(self, i):
            seq, label, reason, is_ep = self.data[i]
            arr = self._augment_seq(seq)
            return torch.tensor(arr, dtype=torch.float32), label, reason, is_ep

    def collate(batch):
        seqs, labels, reasons, is_eps = zip(*batch)
        lengths = torch.tensor([s.size(0) for s in seqs])
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        reasons = torch.tensor(reasons, dtype=torch.long)
        is_endpoints = torch.tensor(is_eps, dtype=torch.float32).unsqueeze(1)
        return padded, lengths, labels, reasons, is_endpoints

    train_ds = SeqDataset(train_samples, augment=True)
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
        SeqDataset(val_samples, augment=False),
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
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=25, verbose=True)

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
    bce = nn.BCEWithLogitsLoss()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0
    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, lengths, y_label, y_reason, y_ep in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            y_label, y_reason = y_label.to(device), y_reason.to(device)
            y_ep = y_ep.to(device)
            opt.zero_grad()
            outputs = model(x, lengths)
            if len(outputs) == 3:
                logits_c, logits_r, logits_ep = outputs
            else:
                logits_c, logits_r = outputs
                logits_ep = None
            loss_ep = bce(logits_ep, y_ep) if logits_ep is not None else x.new_zeros(1)
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
                loss = loss_ep + 0.5 * (loss_c + loss_r)
                train_correct += (logits_c[mask].argmax(1) == y_label[mask]).sum().item()
                train_total += mask.sum().item()
            else:
                loss = loss_ep
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, lengths, y_label, y_reason, y_ep in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                y_label, y_ep = y_label.to(device), y_ep.to(device)
                outputs = model(x, lengths)
                logits_c = outputs[0]
                mask = (y_ep > 0.5).squeeze(1)
                if mask.any():
                    pred = logits_c[mask].argmax(1)
                    val_correct += (pred == y_label[mask]).sum().item()
                    val_total += mask.sum().item()

        val_acc = val_correct / val_total if val_total else 0
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep + 1
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": args.hidden,
                "num_layers": args.lstm_layers,
            }, out_dir / "best.pt")
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": args.hidden,
            "num_layers": args.lstm_layers,
        }, out_dir / "last.pt")
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{args.epochs} loss={train_loss/len(train_loader):.4f} "
                  f"train_acc={train_correct/train_total:.4f} val_acc={val_acc:.4f}")

    print(f"训练完成，最佳 val_acc={best_val_acc:.4f} (epoch {best_epoch})")
    print(f"模型保存: {out_dir / 'best.pt'}, {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()

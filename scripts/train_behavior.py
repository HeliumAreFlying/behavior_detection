"""
训练行为正确性识别模型

支持两种数据源:
  1. track_sequences.json (YOLO 跟踪输出): --data sequences/track_sequences.json
  2. 纯网格 (batch JSON，无需 YOLO): --data grid --batches batches/

用法:
  python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior
  python scripts/train_behavior.py --data grid --batches batches/ -o checkpoints/behavior
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


def load_track_sequences(path: Path, add_velocity: bool = True) -> list[tuple[list[list[float]], int, int]]:
    """
    加载 track_sequences.json，返回 [(features, label_idx, reason_idx), ...]
    features: list of [xc, yc] 或 [xc, yc, dx, dy]
    """
    from models.behavior_correctness import REASON_TO_IDX

    data = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for rec in data:
        feats = rec.get("features", [])
        if len(feats) < 2:
            continue
        # 按 t 排序
        feats = sorted(feats, key=lambda x: x["t"])
        seq = []
        for i, f in enumerate(feats):
            xc, yc = f["xc"], f["yc"]
            if add_velocity and i > 0:
                dx = xc - feats[i - 1]["xc"]
                dy = yc - feats[i - 1]["yc"]
                seq.append([xc, yc, dx, dy])
            else:
                seq.append([xc, yc, 0.0, 0.0] if add_velocity else [xc, yc])
        if not add_velocity:
            seq = [[f["xc"], f["yc"]] for f in feats]

        label = rec.get("label", "correct")
        label_idx = 1 if label == "incorrect" else 0
        reason = rec.get("reason", "in_progress")
        reason_idx = REASON_TO_IDX.get(reason, REASON_TO_IDX["in_progress"])
        samples.append((seq, label_idx, reason_idx))
    return samples


def load_grid_sequences(batches_dir: Path) -> list[tuple[list[list[float]], int, int]]:
    """
    从 batch JSON 加载网格序列: 每蛇每食物周期为一条样本
    features: [head_x_norm, head_y_norm, food_x_norm, food_y_norm, x2_x_norm, x2_y_norm, has_x2, score_norm]
    """
    from models.behavior_correctness import REASON_TO_IDX

    batch_files = sorted(batches_dir.glob("batch_*.json"))
    samples = []
    for bf in batch_files:
        data = json.loads(bf.read_text(encoding="utf-8"))
        for ep in data.get("episodes", []):
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
                samples.append((seq, label_idx, reason_idx))
    return samples


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
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--input-dim", type=int, default=0,
                   help="0=自动: track 用 4，grid 用 8")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-velocity", action="store_true",
                   help="track 数据不加速度特征")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.data.lower() == "grid":
        batches_dir = Path(args.batches)
        if not batches_dir.is_absolute():
            batches_dir = ROOT / batches_dir
        samples = load_grid_sequences(batches_dir)
        input_dim = args.input_dim or 8
        print(f"从 grid 加载: {len(samples)} 条序列, input_dim={input_dim}")
    else:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        if not data_path.exists():
            print(f"文件不存在: {data_path}")
            print("请先运行: python scripts/run_track_and_prepare.py -m <yolo_pt> -d dataset -o sequences")
            sys.exit(1)
        samples = load_track_sequences(data_path, add_velocity=not args.no_velocity)
        input_dim = args.input_dim or (4 if not args.no_velocity else 2)
        print(f"从 track_sequences 加载: {len(samples)} 条序列, input_dim={input_dim}")

    if not samples:
        print("无有效样本")
        sys.exit(1)

    n = len(samples)
    idx = list(range(n))
    random.shuffle(idx)
    val_n = int(n * args.val_ratio)
    train_idx, val_idx = idx[val_n:], idx[:val_n]
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    class SeqDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, i):
            seq, label, reason = self.data[i]
            return torch.tensor(seq, dtype=torch.float32), label, reason

    def collate(batch):
        seqs, labels, reasons = zip(*batch)
        lengths = torch.tensor([s.size(0) for s in seqs])
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        reasons = torch.tensor(reasons, dtype=torch.long)
        return padded, lengths, labels, reasons

    train_loader = DataLoader(
        SeqDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        SeqDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCorrectnessModel(input_dim=input_dim, hidden_dim=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, lengths, y_label, y_reason in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            y_label, y_reason = y_label.to(device), y_reason.to(device)
            opt.zero_grad()
            logits_c, logits_r = model(x, lengths)
            loss = ce(logits_c, y_label) + 0.5 * ce(logits_r, y_reason)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            pred = logits_c.argmax(1)
            train_correct += (pred == y_label).sum().item()
            train_total += y_label.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, lengths, y_label, y_reason in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                y_label = y_label.to(device)
                logits_c, _ = model(x, lengths)
                pred = logits_c.argmax(1)
                val_correct += (pred == y_label).sum().item()
                val_total += y_label.size(0)

        val_acc = val_correct / val_total if val_total else 0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": args.hidden,
            }, out_dir / "best.pt")
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{args.epochs} loss={train_loss/len(train_loader):.4f} "
                  f"train_acc={train_correct/train_total:.4f} val_acc={val_acc:.4f}")

    print(f"训练完成，最佳 val_acc={best_val_acc:.4f}")
    print(f"模型保存: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

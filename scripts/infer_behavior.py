"""
使用训练好的行为正确性模型进行推理

用法:
  python scripts/infer_behavior.py --model checkpoints/behavior/best.pt --data sequences/track_sequences.json
  python scripts/infer_behavior.py -m checkpoints/behavior/best.pt -d grid --batches batches/
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.behavior_correctness import BehaviorCorrectnessModel, REASON_NAMES


def main():
    import argparse
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True, help="模型权重路径")
    p.add_argument("--data", "-d", required=True,
                   help="track_sequences.json 或 grid")
    p.add_argument("--batches", default="batches", help="--data grid 时的 batch 目录")
    p.add_argument("--limit", type=int, default=0, help="最多预测条数，0=全部")
    p.add_argument("--incorrect-threshold", type=float, default=0.5,
                   help="P(incorrect)>=此值则判为错误，降低可提升错误召回率")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    # 兼容旧格式
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        input_dim = ckpt.get("input_dim", 4)
        hidden_dim = ckpt.get("hidden_dim", 64)
        num_layers = ckpt.get("num_layers", 1)
    else:
        state = ckpt
        input_dim, hidden_dim, num_layers = 4, 64, 1

    model = BehaviorCorrectnessModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(state)
    model.eval()

    if args.data.lower() == "grid":
        from train_behavior import load_grid_sequences
        batches_dir = Path(args.batches)
        if not batches_dir.is_absolute():
            batches_dir = ROOT / batches_dir
        samples, _ = load_grid_sequences(batches_dir)
    else:
        from train_behavior import load_track_sequences
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        samples, _ = load_track_sequences(data_path, add_velocity=True)

    if args.limit > 0:
        samples = samples[: args.limit]

    base_dim = 14  # 两种路径统一 14 维（YOLO 可检测的 head/food/x2 及推导特征）
    frame_ctx = input_dim // base_dim if input_dim >= base_dim and input_dim % base_dim == 0 else 1
    half = frame_ctx // 2

    def _merge(seq, bd, h):
        if h <= 0:
            return seq
        n = len(seq)
        return [sum((seq[max(0, min(n - 1, i + j))] for j in range(-h, h + 1)), []) for i in range(n)]

    if half > 0:
        samples = [(_merge(s[0], base_dim, half),) + tuple(s[1:]) for s in samples]

    thresh = args.incorrect_threshold
    correct_pred = 0
    total = 0
    with torch.no_grad():
        for item in samples:
            seq, label_true, reason_true = item[0], item[1], item[2]
            x = torch.tensor([seq], dtype=torch.float32)
            lengths = torch.tensor([len(seq)], dtype=torch.long)
            logits_c, logits_r, _ = model(x, lengths)
            prob_c = torch.softmax(logits_c, dim=1).cpu().numpy()[0]
            pred_c = 1 if prob_c[1] >= thresh else 0
            pred_r = logits_r.argmax(1).item()
            label_pred = "incorrect" if pred_c == 1 else "correct"
            reason_pred = REASON_NAMES[pred_r]
            ok = (1 if pred_c == 1 else 0) == label_true
            correct_pred += ok
            total += 1
            status = "✓" if ok else "✗"
            print(f"{status} pred={label_pred}/{reason_pred} true={'incorrect' if label_true else 'correct'}/{REASON_NAMES[reason_true]}")

    if total:
        print(f"\n准确率: {correct_pred}/{total} = {100*correct_pred/total:.2f}%")


if __name__ == "__main__":
    main()

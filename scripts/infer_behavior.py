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
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    # 兼容旧格式
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        input_dim = ckpt.get("input_dim", 4)
        hidden_dim = ckpt.get("hidden_dim", 64)
    else:
        state = ckpt
        input_dim, hidden_dim = 4, 64

    model = BehaviorCorrectnessModel(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state)
    model.eval()

    if args.data.lower() == "grid":
        from train_behavior import load_grid_sequences
        batches_dir = Path(args.batches)
        if not batches_dir.is_absolute():
            batches_dir = ROOT / batches_dir
        samples = load_grid_sequences(batches_dir)
    else:
        from train_behavior import load_track_sequences
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        samples = load_track_sequences(data_path)

    if args.limit > 0:
        samples = samples[: args.limit]

    correct_pred = 0
    total = 0
    with torch.no_grad():
        for seq, label_true, reason_true in samples:
            x = torch.tensor([seq], dtype=torch.float32)
            lengths = torch.tensor([len(seq)], dtype=torch.long)
            logits_c, logits_r = model(x, lengths)
            pred_c = logits_c.argmax(1).item()
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

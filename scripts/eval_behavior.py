"""
对 batches / track_sequences 进行全量评估，输出 P、R、F1、mAP 等指标（类似 YOLO 评估格式）。

用法:
  # 从 track_sequences 评估（与训练格式完全一致）
  python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json

  # 从 batches 评估（需 dataset metadata 保证帧一致）
  python scripts/eval_behavior.py -c best.pt -b batches/ -d dataset
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GRID_W = GRID_H = 15


def _norm(x: float, size: int) -> float:
    return (x + 0.5) / size


def _scene_to_features(scene: dict) -> dict[int, dict]:
    out = {}
    for si, s in enumerate(scene.get("snakes", [])):
        body = s.get("body", [])
        food = s.get("food", [0, 0])
        x2 = s.get("x2")
        if not body:
            continue
        hx, hy = body[0][0] % GRID_W, body[0][1] % GRID_H
        xc = _norm(hx, GRID_W)
        yc = _norm(hy, GRID_H)
        fx = _norm(int(food[0]) % GRID_W, GRID_W) if food else 0.0
        fy = _norm(int(food[1]) % GRID_H, GRID_H) if food else 0.0
        xx = _norm(int(x2[0]) % GRID_W, GRID_W) if x2 else 0.0
        xy = _norm(int(x2[1]) % GRID_H, GRID_H) if x2 else 0.0
        out[si] = {
            "xc": xc, "yc": yc, "fx": fx, "fy": fy, "xx": xx, "xy": xy,
            "has_x2": 1.0 if x2 else 0.0,
            "x2_active": s.get("x2_active", False),
            "score": s.get("score", 0),
        }
    return out


def _build_seq_features(scene_features: list[dict], input_dim: int) -> dict[int, list[list[float]]]:
    snake_seqs = {}
    num_snakes = max(len(sf) for sf in scene_features) if scene_features else 0
    for si in range(num_snakes):
        seq = []
        for t, sf in enumerate(scene_features):
            if si not in sf:
                continue
            f = sf[si]
            xc, yc = f["xc"], f["yc"]
            fx, fy, xx, xy = f["fx"], f["fy"], f["xx"], f["xy"]
            has_x2 = f["has_x2"]
            x2_active = f.get("x2_active", False)
            score = f.get("score", 0)
            if input_dim == 8:
                feat = [xc, yc, fx, fy, xx, xy, 1.0 if x2_active else 0.0, min(score / 24.0, 1.0)]
            else:
                if t > 0 and si in scene_features[t - 1]:
                    prev = scene_features[t - 1][si]
                    dx, dy = xc - prev["xc"], yc - prev["yc"]
                else:
                    dx, dy = 0.0, 0.0
                df = min(((fx - xc) ** 2 + (fy - yc) ** 2) ** 0.5, 1.5) if (fx or fy) else 0.0
                dx2 = min(((xx - xc) ** 2 + (xy - yc) ** 2) ** 0.5, 1.5) if has_x2 and (xx or xy) else 0.0
                vel = (dx * dx + dy * dy) ** 0.5 or 1e-6
                to_food = (fx - xc) * dx + (fy - yc) * dy
                move_to_food = max(-1, min(1, to_food / vel)) if (fx or fy) else 0.0
                feat = [xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food]
            seq.append(feat)
        if len(seq) >= 2:
            snake_seqs[si] = seq
    return snake_seqs


def load_samples_from_track(path: Path, add_velocity: bool = True):
    """从 track_sequences.json 加载样本，返回 (seq_list, label_list, reason_list)"""
    from models.behavior_correctness import REASON_TO_IDX
    data = json.loads(path.read_text(encoding="utf-8"))
    seqs, labels, reasons = [], [], []
    for rec in data:
        feats = rec.get("features", [])
        if len(feats) < 2:
            continue
        feats = sorted(feats, key=lambda x: x["t"])
        seq = []
        for i, f in enumerate(feats):
            xc, yc = f["xc"], f["yc"]
            fx, fy = f.get("fx", 0.0), f.get("fy", 0.0)
            xx, xy = f.get("xx", 0.0), f.get("xy", 0.0)
            has_x2 = f.get("has_x2", 0.0)
            df = min(((fx - xc) ** 2 + (fy - yc) ** 2) ** 0.5, 1.5) if (fx or fy) else 0.0
            dx2 = min(((xx - xc) ** 2 + (xy - yc) ** 2) ** 0.5, 1.5) if has_x2 and (xx or xy) else 0.0
            if add_velocity and i > 0:
                dx = xc - feats[i - 1]["xc"]
                dy = yc - feats[i - 1]["yc"]
                vel = (dx * dx + dy * dy) ** 0.5 or 1e-6
                to_food = (fx - xc) * dx + (fy - yc) * dy
                move_to_food = max(-1, min(1, to_food / vel)) if (fx or fy) else 0.0
            else:
                dx, dy, move_to_food = 0.0, 0.0, 0.0
            seq.append([xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food])
        seqs.append(seq)
        labels.append(1 if rec.get("label") == "incorrect" else 0)
        reasons.append(REASON_TO_IDX.get(rec.get("reason", "in_progress"), REASON_TO_IDX["in_progress"]))
    return seqs, labels, reasons


def load_samples_from_batches(
    batches_dir: Path, dataset_dir: Path | None, input_dim: int, skip_n: int = 5
):
    """从 batches 加载样本，返回 (seq_list, label_list, reason_list)"""
    from render_and_export import is_key_frame
    from models.behavior_correctness import REASON_TO_IDX

    metadata = None
    if dataset_dir and (dataset_dir / "metadata.json").exists():
        metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        by_ep = defaultdict(list)
        for m in metadata:
            by_ep[(m["batch"], m["episode"])].append(m)
        for k in by_ep:
            by_ep[k].sort(key=lambda x: x["scene"])

    seqs, labels, reasons = [], [], []
    for bf in sorted(batches_dir.glob("batch_*.json")):
        data = json.loads(bf.read_text(encoding="utf-8"))
        for ep_idx, ep in enumerate(data.get("episodes", [])):
            scenes = ep.get("scenes", [])
            if not scenes:
                continue
            anns = ep.get("snake_annotations", [])
            if not anns:
                continue

            if input_dim == 12 and metadata:
                entries = by_ep.get((bf.name, ep_idx), [])
                entries.sort(key=lambda x: x["scene"])
                scene_indices = [e["scene"] for e in entries]
                if not scene_indices:
                    continue
                scene_features = [_scene_to_features(scenes[i]) for i in scene_indices if i < len(scenes)]
            elif input_dim == 12:
                ep_reason = ep.get("reason", "")
                scene_features = []
                for sc_idx, sc in enumerate(scenes):
                    prev = scenes[sc_idx - 1] if sc_idx > 0 else None
                    key = is_key_frame(sc, prev, sc_idx == len(scenes) - 1, ep_reason)
                    if key or skip_n <= 1 or (sc_idx % skip_n == 0):
                        scene_features.append(_scene_to_features(sc))
            else:
                scene_features = [_scene_to_features(sc) for sc in scenes]

            snake_seqs = _build_seq_features(scene_features, input_dim)
            for si, seq in snake_seqs.items():
                if si >= len(anns):
                    continue
                ann = anns[si]
                seqs.append(seq)
                labels.append(1 if ann.get("label") == "incorrect" else 0)
                reasons.append(REASON_TO_IDX.get(ann.get("reason", "in_progress"), REASON_TO_IDX["in_progress"]))

    return seqs, labels, reasons


def main():
    import argparse
    import torch
    import numpy as np

    p = argparse.ArgumentParser(description="行为模型全量评估")
    p.add_argument("-c", "--model", required=True, help="行为模型 checkpoint")
    p.add_argument("-d", "--data", default="", help="track_sequences.json 或 dataset 目录（与 -b 联用）")
    p.add_argument("-b", "--batches", default="", help="batches 目录；指定时从 batch 评估，否则 -d 需为 track_sequences.json")
    p.add_argument("--skip-n", type=int, default=5, help="batches 模式下 track 的跳帧数")
    p.add_argument("--no-velocity", action="store_true", help="track 数据不加速度特征")
    p.add_argument("--max-samples", type=int, default=0, help="最多评估样本数，0=全部")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        input_dim = ckpt.get("input_dim", 12)
        hidden_dim = ckpt.get("hidden_dim", 128)
        num_layers = ckpt.get("num_layers", 2)
    else:
        state, input_dim, hidden_dim, num_layers = ckpt, 12, 128, 2

    from models.behavior_correctness import BehaviorCorrectnessModel, REASON_NAMES

    model = BehaviorCorrectnessModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(state)
    model.eval()

    if args.batches:
        batches_dir = Path(args.batches)
        if not batches_dir.is_absolute():
            batches_dir = ROOT / batches_dir
        dataset_dir = Path(args.data) if args.data else None
        if dataset_dir and not dataset_dir.is_absolute():
            dataset_dir = ROOT / dataset_dir
        print(f"从 batches 加载 (input_dim={input_dim})...")
        seqs, gt_labels, gt_reasons = load_samples_from_batches(
            batches_dir, dataset_dir, input_dim, args.skip_n
        )
    else:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        if not data_path.exists():
            print(f"数据不存在: {data_path}")
            sys.exit(1)
        print(f"从 track_sequences 加载 (input_dim={input_dim})...")
        seqs, gt_labels, gt_reasons = load_samples_from_track(data_path, add_velocity=not args.no_velocity)
        if input_dim == 8:
            print("WARN: track_sequences 通常为 12 维，请确认模型 input_dim")

    if not seqs:
        print("无有效样本")
        sys.exit(1)

    if args.max_samples > 0:
        seqs, gt_labels, gt_reasons = seqs[: args.max_samples], gt_labels[: args.max_samples], gt_reasons[: args.max_samples]
    print(f"共 {len(seqs)} 条样本")

    # 推理
    pred_labels, pred_reasons, pred_reason_probs = [], [], []
    with torch.no_grad():
        for seq in seqs:
            x = torch.tensor([seq], dtype=torch.float32)
            logits_c, logits_r, _ = model(x, None)
            pred_labels.append(logits_c.argmax(1).item())
            pred_reasons.append(logits_r.argmax(1).item())
            pred_reason_probs.append(torch.softmax(logits_r, dim=1).cpu().numpy()[0])

    # 计算指标
    try:
        from sklearn.metrics import (
            precision_recall_fscore_support,
            confusion_matrix,
            accuracy_score,
            average_precision_score,
        )
    except ImportError:
        print("请安装 sklearn: pip install scikit-learn")
        sys.exit(1)

    gt_labels = np.array(gt_labels)
    gt_reasons = np.array(gt_reasons)
    pred_labels = np.array(pred_labels)
    pred_reasons = np.array(pred_reasons)

    # Label (correct=0, incorrect=1)
    print("\n" + "=" * 60)
    print("Label (correct / incorrect)")
    print("=" * 60)
    p_l, r_l, f1_l, _ = precision_recall_fscore_support(gt_labels, pred_labels, average=None, zero_division=0)
    acc_l = accuracy_score(gt_labels, pred_labels)
    for i, name in enumerate(["correct", "incorrect"]):
        print(f"  {name:10s}  P: {p_l[i]:.4f}  R: {r_l[i]:.4f}  F1: {f1_l[i]:.4f}")
    print(f"  {'all':10s}  Accuracy: {acc_l:.4f}")

    # Reason (7 classes)
    print("\n" + "=" * 60)
    print("Reason (各类别)")
    print("=" * 60)
    p_r, r_r, f1_r, sup_r = precision_recall_fscore_support(
        gt_reasons, pred_reasons, labels=range(len(REASON_NAMES)), average=None, zero_division=0
    )
    n_reasons = len(REASON_NAMES)
    # mAP: one-vs-rest 各类 AP 的均值
    pred_probs = np.array(pred_reason_probs)
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(gt_reasons, classes=range(n_reasons))
        ap_per_class = []
        for c in range(n_reasons):
            if y_bin[:, c].sum() == 0:
                ap_per_class.append(float("nan"))
                continue
            ap = average_precision_score(y_bin[:, c], pred_probs[:, c])
            ap_per_class.append(ap)
        valid_ap = [a for a in ap_per_class if not np.isnan(a)]
        mAP = np.mean(valid_ap) if valid_ap else 0.0
    except Exception:
        mAP = 0.0

    # 打印每类
    print(f"  {'class':<22}  {'P':>8}  {'R':>8}  {'F1':>8}  {'support':>8}")
    print("  " + "-" * 56)
    for i in range(n_reasons):
        s = int(sup_r[i]) if i < len(sup_r) else 0
        print(f"  {REASON_NAMES[i]:<22}  {p_r[i]:>8.4f}  {r_r[i]:>8.4f}  {f1_r[i]:>8.4f}  {s:>8}")
    p_macro = np.mean(p_r)
    r_macro = np.mean(r_r)
    f1_macro = np.mean(f1_r)
    print("  " + "-" * 56)
    print(f"  {'macro avg':<22}  {p_macro:>8.4f}  {r_macro:>8.4f}  {f1_macro:>8.4f}")
    print(f"  {'mAP (AP mean)':<22}  {mAP:>8.4f}")

    # 混淆矩阵
    print("\n" + "=" * 60)
    print("Reason 混淆矩阵 (行=GT, 列=Pred)")
    print("=" * 60)
    cm = confusion_matrix(gt_reasons, pred_reasons, labels=range(n_reasons))
    header = "       " + "".join(f"{REASON_NAMES[i][:6]:>8}" for i in range(n_reasons))
    print(header)
    for i in range(n_reasons):
        row = f"{REASON_NAMES[i][:6]:>6}" + "".join(f"{cm[i,j]:>8}" for j in range(n_reasons))
        print(row)


if __name__ == "__main__":
    main()

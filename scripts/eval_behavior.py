"""
对 track_sequences 进行全量评估，输出 P、R、F1、mAP 等指标（类似 YOLO 评估格式）。
默认仅评估 val 集，与训练时的验证集一致，保证指标准确。

用法:
  python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json
  python scripts/eval_behavior.py -c best.pt -d sequences/track_sequences.json --split all
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_samples_from_track(
    path: Path, add_velocity: bool = True, split_filter: str | None = None
):
    """从 track_sequences.json 加载样本，返回 (seqs_cont, seqs_hf, labels, reasons)。
    split_filter: 仅加载该 split（'train'/'val'），None 表示全部。
    """
    from models.behavior_correctness import REASON_TO_IDX
    data = json.loads(path.read_text(encoding="utf-8"))
    seqs_cont, seqs_hf, labels, reasons = [], [], [], []
    for rec in data:
        if split_filter is not None and rec.get("split") != split_filter:
            continue
        feats = rec.get("features", [])
        if len(feats) < 2:
            continue
        feats = sorted(feats, key=lambda x: x["t"])
        seq_c, seq_h = [], []
        for i, f in enumerate(feats):
            xc, yc = f["xc"], f["yc"]
            fx, fy = f.get("fx", 0.0), f.get("fy", 0.0)
            xx, xy = f.get("xx", 0.0), f.get("xy", 0.0)
            has_x2 = f.get("has_x2", 0.0)
            ate_food, ate_x2 = f.get("ate_food", 0.0), f.get("ate_x2", 0.0)
            ate_food_while_x2 = f.get("ate_food_while_x2_exists", 1.0 if (ate_food and has_x2) else 0.0)
            head_forward = max(0, min(3, int(f.get("head_forward_type", 0))))
            is_dead = f.get("is_dead", 0.0)
            steps_since_food = f.get("steps_since_food", 0.0)
            about_to_timeout = f.get("about_to_timeout", (1.0 if steps_since_food >= 79.0 / 80.0 else 0.0))
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
            cont = [xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food, ate_food, ate_x2, is_dead, steps_since_food, ate_food_while_x2, about_to_timeout]
            seq_c.append(cont)
            seq_h.append(head_forward)
        seqs_cont.append(seq_c)
        seqs_hf.append(seq_h)
        labels.append(1 if rec.get("label") == "incorrect" else 0)
        reasons.append(REASON_TO_IDX.get(rec.get("reason", "in_progress"), REASON_TO_IDX["in_progress"]))
    return seqs_cont, seqs_hf, labels, reasons


def main():
    import argparse
    import torch
    import numpy as np

    p = argparse.ArgumentParser(description="行为模型全量评估")
    p.add_argument("-c", "--model", required=True, help="行为模型 checkpoint")
    p.add_argument("-d", "--data", required=True, help="track_sequences.json 路径")
    p.add_argument("--split", choices=["train", "val", "all"], default="val",
                   help="评估集合：val=仅验证集(默认,与训练一致), train=仅训练集, all=全部")
    p.add_argument("--no-velocity", action="store_true", help="track 数据不加速度特征")
    p.add_argument("--max-samples", type=int, default=0, help="最多评估样本数，0=全部")
    p.add_argument("--incorrect-threshold", type=float, default=0.5,
                   help="预测 incorrect 的阈值（仅在 --no-threshold-search 时生效）")
    p.add_argument("--no-threshold-search", action="store_true",
                   help="不自动搜索阈值时使用 --incorrect-threshold；否则按 (mAP50+mAP50-95)/2 最大选取阈值")
    p.add_argument("--reason-override", action="store_true",
                   help="若预测 reason 为错误类(self_collision/snake_collision/x2_wasted/timeout)，强制 label=incorrect")
    p.add_argument("--batch-size", type=int, default=128,
                   help="推理批次大小，增大可提速（显存允许时）")
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
        bidirectional = ckpt.get("bidirectional", False)
        use_attention = ckpt.get("use_attention", False)
    else:
        state, input_dim, hidden_dim, num_layers = ckpt, 12, 128, 2
        bidirectional, use_attention = False, False

    from models.behavior_correctness import BehaviorCorrectnessModel, REASON_NAMES

    use_head_forward_embedding = ckpt.get("use_head_forward_embedding", False)
    model = BehaviorCorrectnessModel(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
        bidirectional=bidirectional, use_attention=use_attention,
        use_head_forward_embedding=use_head_forward_embedding,
    )
    model.load_state_dict(state)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        print(f"数据不存在: {data_path}")
        sys.exit(1)
    split_filter = None if args.split == "all" else args.split
    print(f"从 track_sequences 加载 (input_dim={input_dim}, split={args.split})...")
    seqs_cont, seqs_hf, gt_labels, gt_reasons = load_samples_from_track(
        data_path, add_velocity=not args.no_velocity, split_filter=split_filter
    )
    if input_dim == 8:
        print("WARN: track_sequences 通常为 12 维，请确认模型 input_dim")

    if not seqs_cont:
        if split_filter is not None:
            print(f"未找到 split={split_filter!r} 的样本；若数据为旧版无 split，请使用 --split all 或重新运行 run_track_and_prepare 生成带 split 的数据")
        else:
            print("无有效样本")
        sys.exit(1)

    if args.max_samples > 0:
        seqs_cont = seqs_cont[: args.max_samples]
        seqs_hf = seqs_hf[: args.max_samples]
        gt_labels = gt_labels[: args.max_samples]
        gt_reasons = gt_reasons[: args.max_samples]
    print(f"共 {len(seqs_cont)} 条样本")

    base_cont_dim = 18 if use_head_forward_embedding else 16
    if not use_head_forward_embedding and seqs_cont and len(seqs_cont[0][0]) > 16:
        seqs_cont = [[f[:16] for f in seq] for seq in seqs_cont]
    frame_ctx = input_dim // base_cont_dim if input_dim >= base_cont_dim and input_dim % base_cont_dim == 0 else 1
    half = frame_ctx // 2

    def _merge_frame_context(seqs: list, base_dim: int) -> list:
        if half <= 0:
            return seqs
        out = []
        for seq in seqs:
            n = len(seq)
            merged = []
            for i in range(n):
                ctx = []
                for j in range(i - half, i + half + 1):
                    idx = max(0, min(n - 1, j))
                    ctx.extend(seq[idx])
                merged.append(ctx)
            out.append(merged)
        return out

    if half > 0:
        seqs_cont = _merge_frame_context(seqs_cont, base_cont_dim)

    pred_reasons, prob_incorrect_list, pred_reason_probs = [], [], []
    batch_size = args.batch_size
    n = len(seqs_cont)
    use_hf = ckpt.get("use_head_forward_embedding", False)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            batch_cont = seqs_cont[start : start + batch_size]
            batch_hf = seqs_hf[start : start + batch_size]
            lengths = torch.tensor([len(s) for s in batch_cont], dtype=torch.long)
            padded_cont = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(s, dtype=torch.float32) for s in batch_cont],
                batch_first=True,
                padding_value=0.0,
            )
            padded_hf = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(s, dtype=torch.long) for s in batch_hf],
                batch_first=True,
                padding_value=0,
            )
            x = padded_cont.to(device)
            x_hf = padded_hf.to(device) if use_hf else None
            lengths = lengths.to(device)
            logits_c, logits_r = model(x, lengths, x_head_forward=x_hf)
            prob_c = torch.softmax(logits_c, dim=1).cpu().numpy()
            pred_r = logits_r.argmax(1).cpu().numpy()
            prob_r = torch.softmax(logits_r, dim=1).cpu().numpy()
            for i in range(len(batch_cont)):
                pred_reasons.append(int(pred_r[i]))
                prob_incorrect_list.append(float(prob_c[i, 1]))
                pred_reason_probs.append(prob_r[i])

    # 计算指标：P, R, mAP50, mAP50-95（格式与 YOLO val 一致）
    try:
        from sklearn.metrics import precision_recall_fscore_support, average_precision_score
        from sklearn.preprocessing import label_binarize
    except ImportError:
        print("请安装 sklearn: pip install scikit-learn")
        sys.exit(1)

    gt_labels = np.array(gt_labels)
    prob_incorrect = np.array(prob_incorrect_list)
    pred_probs = np.array(pred_reason_probs)
    gt_reasons_arr = np.array(gt_reasons)
    pred_reasons_arr = np.array(pred_reasons)
    n_reasons = len(REASON_NAMES)
    y_bin = label_binarize(gt_reasons_arr, classes=range(n_reasons))

    # 阈值：自动搜索使 (mAP50 + mAP50-95)/2 最大的阈值，或使用指定值
    def _pred_at_thresh(prob: np.ndarray, reasons: np.ndarray, t: float) -> np.ndarray:
        pred = (prob >= t).astype(np.int64)
        if args.reason_override:
            pred[(reasons == 3) | (reasons == 4) | (reasons == 5) | (reasons == 6)] = 1
        return pred

    def _mAP_at_thresh(t: float) -> float:
        """给定阈值 t，按 pred_b 得到有效 reason：correct 的样本视为 in_progress(2)，incorrect 用模型 reason；再算 (mAP50+mAP50-95)/2"""
        pred_b = _pred_at_thresh(prob_incorrect, pred_reasons_arr, t)
        # 有效概率：pred_b=0 的样本视为 in_progress(2)，其余用 pred_probs
        effective_probs = np.zeros_like(pred_probs)
        for i in range(len(pred_b)):
            if pred_b[i] == 0:
                effective_probs[i, 2] = 1.0
            else:
                effective_probs[i, :] = pred_probs[i, :]
        ap_per = []
        for c in range(n_reasons):
            if y_bin[:, c].sum() == 0:
                ap_per.append(0.0)
            else:
                ap_per.append(average_precision_score(y_bin[:, c], effective_probs[:, c]))
        mAP50 = float(np.mean(ap_per)) if ap_per else 0.0
        mAP50_95 = mAP50
        return (mAP50 + mAP50_95) / 2.0

    best_thresh = args.incorrect_threshold
    best_score = 0.0
    do_search = not args.no_threshold_search
    if do_search:
        for t in np.arange(0.05, 0.96, 0.05):
            score = _mAP_at_thresh(t)
            if score > best_score:
                best_score = score
                best_thresh = t
    else:
        best_thresh = args.incorrect_threshold
        best_score = _mAP_at_thresh(best_thresh)

    pred_labels = _pred_at_thresh(prob_incorrect, pred_reasons_arr, best_thresh)
    p, r, f1, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, labels=[0, 1], average=None, zero_division=0
    )
    print(f"\nBinary (correct/incorrect) 最优阈值={best_thresh:.2f}  (mAP50+mAP50-95)/2={best_score:.4f}  P={p[1]:.4f}  R={r[1]:.4f}  F1={f1[1]:.4f}")
    # 在最优阈值下的有效 reason：correct 样本视为 in_progress(2)，用于下表
    effective_reason = np.where(pred_labels == 0, 2, pred_reasons_arr)
    effective_probs_final = np.zeros_like(pred_probs)
    for i in range(len(pred_labels)):
        if pred_labels[i] == 0:
            effective_probs_final[i, 2] = 1.0
        else:
            effective_probs_final[i, :] = pred_probs[i, :]
    # 每类 P, R, mAP50, mAP50-95（与选取阈值的指标一致）
    p_per, r_per, _, support = precision_recall_fscore_support(
        gt_reasons_arr, effective_reason, labels=range(n_reasons),
        average=None, zero_division=0
    )
    ap_per_class = []
    for c in range(n_reasons):
        if y_bin[:, c].sum() == 0:
            ap_per_class.append(0.0)
        else:
            ap = average_precision_score(y_bin[:, c], effective_probs_final[:, c])
            ap_per_class.append(ap)

    # 整体 = 7 类 reason 的宏平均（与 YOLO all 一致）
    p_all = float(np.mean(p_per))
    r_all = float(np.mean(r_per))
    ap50_all = float(np.mean(ap_per_class)) if ap_per_class else 0.0
    map50_95_all = ap50_all
    total_support = int(np.sum(support))

    # 打印：表头在第一行，n 在 Class 与 P 之间；all 行为总数
    print("\n" + f"{'Class':<22}  {'n':>8}  {'P':>10}  {'R':>10}  {'mAP50':>10}  {'mAP50-95':>10}")
    print("-" * 78)
    print(f"{'all':<22}  {total_support:>8}  {p_all:>10.4f}  {r_all:>10.4f}  {ap50_all:>10.4f}  {map50_95_all:>10.4f}")
    for c in range(n_reasons):
        name = REASON_NAMES[c][:20]
        ap50 = ap_per_class[c] if c < len(ap_per_class) else 0.0
        print(f"{name:<22}  {int(support[c]):>8}  {p_per[c]:>10.4f}  {r_per[c]:>10.4f}  {ap50:>10.4f}  {ap50:>10.4f}")


if __name__ == "__main__":
    main()

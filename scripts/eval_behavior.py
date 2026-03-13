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


def _head_forward_type_from_scene(scene: dict) -> list[int]:
    """蛇头前方格子: 0=空, 1=自己身体, 2=其他蛇"""
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
        other_cells = set()
        for sj, s2 in enumerate(snakes_data):
            if sj == si:
                continue
            for p in s2.get("body", []):
                try:
                    gx = (int(p[0]) % GRID_W + GRID_W) % GRID_W
                    gy = (int(p[1]) % GRID_H + GRID_H) % GRID_H
                    other_cells.add((gx, gy))
                except (IndexError, TypeError):
                    pass
        fwd = (fx, fy)
        if fwd in own_cells:
            result.append(1)
        elif fwd in other_cells:
            result.append(2)
        else:
            result.append(0)
    return result


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


def _infer_ate_from_scene(prev: dict | None, curr: dict) -> tuple[float, float]:
    """从前后帧 scene 特征推断 ate_food, ate_x2"""
    if prev is None:
        return 0.0, 0.0
    xc, yc = curr["xc"], curr["yc"]
    fx, fy = curr["fx"], curr["fy"]
    thresh = 0.02
    def _d(ax, ay, bx, by): return (ax - bx) ** 2 + (ay - by) ** 2
    ate_food = 0.0
    pfx, pfy = prev["fx"], prev["fy"]
    if (pfx or pfy) and _d(fx, fy, pfx, pfy) > thresh and _d(xc, yc, pfx, pfy) < thresh:
        ate_food = 1.0
    ate_x2 = 0.0
    if prev.get("has_x2") and (not curr.get("has_x2") or _d(curr["xx"], curr["xy"], prev["xx"], prev["xy"]) > thresh):
        pxx, pxy = prev["xx"], prev["xy"]
        if (pxx or pxy) and _d(xc, yc, pxx, pxy) < thresh:
            ate_x2 = 1.0
    return ate_food, ate_x2


def _build_seq_features(
    scene_features: list[dict],
    scenes: list[dict] | None,
    input_dim: int,
    last_frame_reasons: dict[int, str] | None = None,
) -> dict[int, tuple[list[list[float]], list[int]]]:
    """构建 17 维 cont + head_forward。scenes 用于计算 head_forward，None 时用 0。"""
    snake_seqs: dict[int, tuple[list[list[float]], list[int]]] = {}
    num_snakes = max(len(sf) for sf in scene_features) if scene_features else 0
    last_t_per_snake: dict[int, int] = {}
    for t, sf in enumerate(scene_features):
        for si in sf:
            last_t_per_snake[si] = t
    for si in range(num_snakes):
        seq_cont, seq_hf = [], []
        steps_counter = 0
        is_dead_last = 0.0
        if last_frame_reasons and si in last_frame_reasons:
            r = last_frame_reasons[si]
            if r in ("self_collision", "snake_collision"):
                is_dead_last = 1.0
        for t, sf in enumerate(scene_features):
            if si not in sf:
                continue
            f = sf[si]
            xc, yc = f["xc"], f["yc"]
            fx, fy, xx, xy = f["fx"], f["fy"], f["xx"], f["xy"]
            has_x2 = f["has_x2"]
            prev_f = scene_features[t - 1][si] if t > 0 and si in scene_features[t - 1] else None
            ate_food, ate_x2 = _infer_ate_from_scene(prev_f, f)
            ate_food_while_x2 = 1.0 if (ate_food and has_x2) else 0.0
            head_forward = 0
            if scenes and t < len(scenes):
                hf_list = _head_forward_type_from_scene(scenes[t])
                head_forward = int(hf_list[si]) if si < len(hf_list) else 0
                head_forward = max(0, min(2, head_forward))
            if ate_food:
                steps_counter = 0
            else:
                steps_counter += 1
            steps_since_food = min(steps_counter / 80.0, 1.0)
            is_dead = is_dead_last if last_t_per_snake.get(si, -1) == t else 0.0
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
            cont = [xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food, ate_food, ate_x2, is_dead, steps_since_food, ate_food_while_x2]
            seq_cont.append(cont)
            seq_hf.append(head_forward)
        if len(seq_cont) >= 2:
            snake_seqs[si] = (seq_cont, seq_hf)
    return snake_seqs


def load_samples_from_track(path: Path, add_velocity: bool = True):
    """从 track_sequences.json 加载样本，返回 (seqs_cont, seqs_hf, labels, reasons)。17 维 cont + head_forward。"""
    from models.behavior_correctness import REASON_TO_IDX
    data = json.loads(path.read_text(encoding="utf-8"))
    seqs_cont, seqs_hf, labels, reasons = [], [], [], []
    for rec in data:
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
            head_forward = max(0, min(2, int(f.get("head_forward_type", 0))))
            is_dead = f.get("is_dead", 0.0)
            steps_since_food = f.get("steps_since_food", 0.0)
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
            cont = [xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food, ate_food, ate_x2, is_dead, steps_since_food, ate_food_while_x2]
            seq_c.append(cont)
            seq_h.append(head_forward)
        seqs_cont.append(seq_c)
        seqs_hf.append(seq_h)
        labels.append(1 if rec.get("label") == "incorrect" else 0)
        reasons.append(REASON_TO_IDX.get(rec.get("reason", "in_progress"), REASON_TO_IDX["in_progress"]))
    return seqs_cont, seqs_hf, labels, reasons


def load_samples_from_batches(
    batches_dir: Path, dataset_dir: Path | None, input_dim: int
):
    """从 batches 加载样本，返回 (seq_list, label_list, reason_list)。不跳帧，使用全帧。"""
    from models.behavior_correctness import REASON_TO_IDX

    metadata = None
    if dataset_dir and (dataset_dir / "metadata.json").exists():
        metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        by_ep = defaultdict(list)
        for m in metadata:
            by_ep[(m["batch"], m["episode"])].append(m)
        for k in by_ep:
            by_ep[k].sort(key=lambda x: x["scene"])

    seqs_cont, seqs_hf, labels, reasons = [], [], [], []
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
                scenes_sub = [scenes[i] for i in scene_indices if i < len(scenes)]
            else:
                scene_features = [_scene_to_features(sc) for sc in scenes]
                scenes_sub = scenes

            last_reasons = {si: anns[si].get("reason", "in_progress") for si in range(len(anns)) if si < len(anns)}
            snake_seqs = _build_seq_features(scene_features, scenes_sub, input_dim, last_frame_reasons=last_reasons)
            for si, (seq_c, seq_h) in snake_seqs.items():
                if si >= len(anns):
                    continue
                ann = anns[si]
                seqs_cont.append(seq_c)
                seqs_hf.append(seq_h)
                labels.append(1 if ann.get("label") == "incorrect" else 0)
                reasons.append(REASON_TO_IDX.get(ann.get("reason", "in_progress"), REASON_TO_IDX["in_progress"]))

    return seqs_cont, seqs_hf, labels, reasons


def main():
    import argparse
    import torch
    import numpy as np

    p = argparse.ArgumentParser(description="行为模型全量评估")
    p.add_argument("-c", "--model", required=True, help="行为模型 checkpoint")
    p.add_argument("-d", "--data", default="", help="track_sequences.json 或 dataset 目录（与 -b 联用）")
    p.add_argument("-b", "--batches", default="", help="batches 目录；指定时从 batch 评估，否则 -d 需为 track_sequences.json")
    p.add_argument("--no-velocity", action="store_true", help="track 数据不加速度特征")
    p.add_argument("--max-samples", type=int, default=0, help="最多评估样本数，0=全部")
    p.add_argument("--incorrect-threshold", type=float, default=0.5,
                   help="预测 incorrect 的阈值（仅在 --no-threshold-search 时生效）")
    p.add_argument("--no-threshold-search", action="store_true",
                   help="不自动搜索阈值，使用 --incorrect-threshold 指定值")
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

    if args.batches:
        batches_dir = Path(args.batches)
        if not batches_dir.is_absolute():
            batches_dir = ROOT / batches_dir
        dataset_dir = Path(args.data) if args.data else None
        if dataset_dir and not dataset_dir.is_absolute():
            dataset_dir = ROOT / dataset_dir
        print(f"从 batches 加载 (input_dim={input_dim})...")
        seqs_cont, seqs_hf, gt_labels, gt_reasons = load_samples_from_batches(
            batches_dir, dataset_dir, input_dim
        )
    else:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        if not data_path.exists():
            print(f"数据不存在: {data_path}")
            sys.exit(1)
        print(f"从 track_sequences 加载 (input_dim={input_dim})...")
        seqs_cont, seqs_hf, gt_labels, gt_reasons = load_samples_from_track(data_path, add_velocity=not args.no_velocity)
        if input_dim == 8:
            print("WARN: track_sequences 通常为 12 维，请确认模型 input_dim")

    if not seqs_cont:
        print("无有效样本")
        sys.exit(1)

    if args.max_samples > 0:
        seqs_cont = seqs_cont[: args.max_samples]
        seqs_hf = seqs_hf[: args.max_samples]
        gt_labels = gt_labels[: args.max_samples]
        gt_reasons = gt_reasons[: args.max_samples]
    print(f"共 {len(seqs_cont)} 条样本")

    base_cont_dim = 17 if use_head_forward_embedding else 16
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

    # 阈值：自动搜索最优 F1 或使用指定值
    def _pred_at_thresh(prob: np.ndarray, reasons: np.ndarray, t: float) -> np.ndarray:
        pred = (prob >= t).astype(np.int64)
        if args.reason_override:
            pred[(reasons == 3) | (reasons == 4) | (reasons == 5) | (reasons == 6)] = 1
        return pred

    best_thresh = args.incorrect_threshold
    best_f1 = 0.0
    best_p = best_r = 0.0
    do_search = not args.no_threshold_search
    if do_search:
        for t in np.arange(0.05, 0.96, 0.05):
            pred_b = _pred_at_thresh(prob_incorrect, pred_reasons_arr, t)
            p, r, f1, _ = precision_recall_fscore_support(
                gt_labels, pred_b, labels=[0, 1], average=None, zero_division=0
            )
            if gt_labels.sum() > 0 and f1[1] > best_f1:
                best_f1 = f1[1]
                best_thresh = t
                best_p, best_r = p[1], r[1]
    else:
        pred_b = _pred_at_thresh(prob_incorrect, pred_reasons_arr, args.incorrect_threshold)
        p, r, f1, _ = precision_recall_fscore_support(
            gt_labels, pred_b, labels=[0, 1], average=None, zero_division=0
        )
        best_thresh = args.incorrect_threshold
        best_p, best_r, best_f1 = p[1], r[1], f1[1]

    pred_labels = _pred_at_thresh(prob_incorrect, pred_reasons_arr, best_thresh)
    print(f"\nBinary (correct/incorrect) 最优阈值={best_thresh:.2f}  P={best_p:.4f}  R={best_r:.4f}  F1={best_f1:.4f}")
    # 每类 P, R, mAP50, mAP50-95
    p_per, r_per, _, support = precision_recall_fscore_support(
        gt_reasons_arr, pred_reasons_arr, labels=range(n_reasons),
        average=None, zero_division=0
    )
    ap_per_class = []
    for c in range(n_reasons):
        if y_bin[:, c].sum() == 0:
            ap_per_class.append(0.0)
        else:
            ap = average_precision_score(y_bin[:, c], pred_probs[:, c])
            ap_per_class.append(ap)

    # 整体 = 7 类 reason 的宏平均（与 YOLO all 一致）
    p_all = float(np.mean(p_per))
    r_all = float(np.mean(r_per))
    ap50_all = float(np.mean(ap_per_class)) if ap_per_class else 0.0
    map50_95_all = ap50_all

    # 打印：表头在第一行，与 YOLO 一致
    print("\n" + f"{'Class':<22}  {'P':>10}  {'R':>10}  {'mAP50':>10}  {'mAP50-95':>10}")
    print("-" * 68)
    print(f"{'all':<22}  {p_all:>10.4f}  {r_all:>10.4f}  {ap50_all:>10.4f}  {map50_95_all:>10.4f}")
    for c in range(n_reasons):
        name = REASON_NAMES[c][:20]
        ap50 = ap_per_class[c] if c < len(ap_per_class) else 0.0
        print(f"{name:<22}  {p_per[c]:>10.4f}  {r_per[c]:>10.4f}  {ap50:>10.4f}  {ap50:>10.4f}")


if __name__ == "__main__":
    main()

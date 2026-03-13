"""
实战测试：将对局记录渲染为视频，YOLO+行为模型联合标注，并同时展示真值（GT）以对比准确率。

注意：track 模型需与训练帧一致。推荐 -d dataset 从 metadata 读取实际导出帧。

用法:
  python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c checkpoints/behavior/best.pt -o demo.mp4
  python scripts/demo_video.py -b batches/batch_00000.json -e 0 -m yolov8n.pt -c checkpoints/behavior/best.pt -o demo.mp4 --draw-boxes
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# 无头环境下抑制 pygame/SDL 的 ALSA 音频警告（不影响渲染）
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GRID_W = GRID_H = 15
IMG_W = IMG_H = 640


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


def _norm(x: float, size: int) -> float:
    """网格坐标归一化 [0,1]"""
    return (x + 0.5) / size


def _scene_to_features(scene: dict) -> dict[int, dict]:
    """从 scene 提取每蛇特征，与 train_behavior 保持一致"""
    out: dict[int, dict] = {}
    snakes = scene.get("snakes", [])
    for si, s in enumerate(snakes):
        body = s.get("body", [])
        food = s.get("food", [0, 0])
        x2 = s.get("x2")
        score = s.get("score", 0)
        x2_active = s.get("x2_active", False)
        if not body:
            continue
        hx, hy = body[0][0] % GRID_W, body[0][1] % GRID_H
        xc = _norm(hx, GRID_W)
        yc = _norm(hy, GRID_H)
        fx = _norm(int(food[0]) % GRID_W, GRID_W) if food else 0.0
        fy = _norm(int(food[1]) % GRID_H, GRID_H) if food else 0.0
        xx = _norm(int(x2[0]) % GRID_W, GRID_W) if x2 else 0.0
        xy = _norm(int(x2[1]) % GRID_H, GRID_H) if x2 else 0.0
        has_x2 = 1.0 if x2 else 0.0
        out[si] = {"xc": xc, "yc": yc, "fx": fx, "fy": fy, "xx": xx, "xy": xy, "has_x2": has_x2, "x2_active": x2_active, "score": score}
    return out


def _infer_ate_from_scene(prev: dict | None, curr: dict) -> tuple[float, float]:
    if prev is None:
        return 0.0, 0.0
    xc, yc = curr["xc"], curr["yc"]
    thresh = 0.02
    def _d(ax, ay, bx, by): return (ax - bx) ** 2 + (ay - by) ** 2
    ate_food = 0.0
    pfx, pfy, fx, fy = prev["fx"], prev["fy"], curr["fx"], curr["fy"]
    if (pfx or pfy) and _d(fx, fy, pfx, pfy) > thresh and _d(xc, yc, pfx, pfy) < thresh:
        ate_food = 1.0
    ate_x2 = 0.0
    if prev.get("has_x2") and (not curr.get("has_x2") or _d(curr["xx"], curr["xy"], prev["xx"], prev["xy"]) > thresh):
        pxx, pxy = prev["xx"], prev["xy"]
        if (pxx or pxy) and _d(xc, yc, pxx, pxy) < thresh:
            ate_x2 = 1.0
    return ate_food, ate_x2


def _build_seq_features(
    scene_features: list[dict[int, dict]],
    scenes: list[dict] | None,
    input_dim: int = 17,
    last_frame_reasons: dict[int, str] | None = None,
) -> dict[int, tuple[list[list[float]], list[int]]]:
    """构建 17 维 cont + head_forward。"""
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


def main():
    import argparse
    import numpy as np
    import torch

    p = argparse.ArgumentParser(description="YOLO+行为模型联合标注视频")
    p.add_argument("--batch", "-b", required=True, help="batch JSON 路径")
    p.add_argument("--episode", "-e", type=int, default=0, help="episode 索引")
    p.add_argument("--yolo-model", "-m", default="", help="YOLO 权重路径；不指定则跳过 YOLO 跟踪与框绘制")
    p.add_argument("--behavior-model", "-c", required=True, help="行为模型 best.pt 路径")
    p.add_argument("--output", "-o", default="demo_annotated.mp4", help="输出视频路径")
    p.add_argument("--fps", type=int, default=8, help="视频帧率")
    p.add_argument("--draw-boxes", action="store_true", help="绘制 YOLO 检测框")
    p.add_argument("--label-only", action="store_true", help="仅按 correct/incorrect 判定一致，不要求 reason 相同")
    p.add_argument("--dataset", "-d", default="",
                   help="dataset 目录；指定后从 metadata 读取实际导出帧，保证与训练完全一致")
    args = p.parse_args()

    try:
        import pygame
        import cv2
    except ImportError as e:
        print(f"请安装依赖: pip install pygame opencv-python\n{e}")
        sys.exit(1)

    from render_and_export import render_scene, scene_to_bboxes
    from models.behavior_correctness import BehaviorCorrectnessModel, REASON_NAMES
    from replay_ui import REASON_NAMES as REASON_ZH

    batch_path = Path(args.batch)
    if not batch_path.is_absolute():
        batch_path = ROOT / batch_path
    if not batch_path.exists():
        print(f"文件不存在: {batch_path}")
        sys.exit(1)

    data = json.loads(batch_path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    if args.episode >= len(episodes):
        print(f"episode {args.episode} 不存在，共 {len(episodes)} 个")
        sys.exit(1)

    ep = episodes[args.episode]
    scenes = ep.get("scenes", [])
    if not scenes:
        print("该 episode 无场景")
        sys.exit(1)

    # 真值（batch 中的 snake_annotations）
    gt_annotations = ep.get("snake_annotations", [])

    pygame.init()
    pygame.display.set_mode((1, 1))

    # 渲染帧（蛇头：开局菱形→三角→撞击圆形）
    frames_np = []
    scene_features = []
    for ti, sc in enumerate(scenes):
        prev_sc = scenes[ti - 1] if ti > 0 else None
        surf = render_scene(sc, scene_idx=ti, prev_scene=prev_sc, total_scenes=len(scenes))
        arr = np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2))
        frames_np.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        scene_features.append(_scene_to_features(sc))

    # YOLO track（可选）
    results_list: list = []
    if args.yolo_model:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo_model)
        results_list = list(yolo.track(
            source=frames_np,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            stream=True,
        ))

    # 行为模型
    ckpt_path = Path(args.behavior_model)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        print(f"行为模型不存在: {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    use_head_forward_embedding = ckpt.get("use_head_forward_embedding", False) if isinstance(ckpt, dict) else False

    beh_model = BehaviorCorrectnessModel(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
        bidirectional=bidirectional, use_attention=use_attention,
        use_head_forward_embedding=use_head_forward_embedding,
    )
    beh_model.load_state_dict(state)
    beh_model.eval()

    use_metadata = False
    scene_indices = []
    if input_dim >= 14:
        dataset_dir = Path(args.dataset) if args.dataset else (ROOT / "dataset")
        if not dataset_dir.is_absolute():
            dataset_dir = ROOT / dataset_dir
        meta_path = dataset_dir / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            entries = [m for m in metadata if m.get("batch") == batch_path.name and m.get("episode") == args.episode]
            entries.sort(key=lambda x: x["scene"])
            scene_indices = [e["scene"] for e in entries]
            if scene_indices:
                scene_features_for_model = [_scene_to_features(scenes[i]) for i in scene_indices if i < len(scenes)]
                print(f"行为模型 input_dim={input_dim} (track), 从 dataset metadata 读取 {len(scene_features_for_model)} 帧")
                use_metadata = True
        if not use_metadata:
            scene_features_for_model = scene_features
            print(f"行为模型 input_dim={input_dim} (track), 全帧 {len(scene_features_for_model)} 帧 (建议 -d dataset 与训练一致)")
    else:
        scene_features_for_model = scene_features
        print(f"行为模型 input_dim={input_dim}, 全帧 {len(scene_features)} 帧")
    base_cont_dim = 17 if use_head_forward_embedding else 16
    scenes_for_model = [scenes[i] for i in scene_indices if i < len(scenes)] if use_metadata and scene_indices else scenes
    last_reasons = {si: gt_annotations[si].get("reason", "in_progress") for si in range(len(gt_annotations))}
    snake_seqs = _build_seq_features(scene_features_for_model, scenes_for_model, base_cont_dim, last_frame_reasons=last_reasons)

    def _merge_frame_context(seq: list[list[float]], base: int, half: int) -> list[list[float]]:
        if half <= 0:
            return seq
        n = len(seq)
        return [
            sum((seq[max(0, min(n - 1, i + j))] for j in range(-half, half + 1)), [])
            for i in range(n)
        ]

    frame_ctx = input_dim // base_cont_dim if input_dim >= base_cont_dim and input_dim % base_cont_dim == 0 else 1
    half = frame_ctx // 2
    if half > 0:
        snake_seqs = {si: (_merge_frame_context(seq_c, base_cont_dim, half), seq_h) for si, (seq_c, seq_h) in snake_seqs.items()}

    snake_preds: dict[int, list[tuple[int, str, str, float]]] = defaultdict(list)
    for si, (seq_c, seq_h) in snake_seqs.items():
        with torch.no_grad():
            x = torch.tensor([seq_c], dtype=torch.float32)
            x_hf = torch.tensor([seq_h], dtype=torch.long) if use_head_forward_embedding else None
            logits_c, logits_r = beh_model(x, None, x_head_forward=x_hf)
            pred_c = logits_c.argmax(1).item()
            pred_r = logits_r.argmax(1).item()
        label = "incorrect" if pred_c == 1 else "correct"
        reason = REASON_NAMES[pred_r]
        snake_preds[si] = [(len(seq_c) - 1, label, reason)]

    # 每帧展示：GT + 预测，用于对比
    labels_per_frame: dict[int, dict[int, tuple[str, str, str, str, bool]]] = defaultdict(dict)
    for ti in range(len(scenes)):
        for si in scene_features[ti]:
            gt_label = gt_annotations[si]["label"] if si < len(gt_annotations) else "?"
            gt_reason = gt_annotations[si]["reason"] if si < len(gt_annotations) else "?"
            if si in snake_preds:
                _, pred_label, pred_reason = snake_preds[si][0]
                match = (gt_label == pred_label and gt_reason == pred_reason) if not args.label_only else (gt_label == pred_label)
            else:
                pred_label, pred_reason, match = "-", "-", False
            labels_per_frame[ti][si] = (gt_label, gt_reason, pred_label, pred_reason, match)

    # 绘制并写视频
    h, w = frames_np[0].shape[:2]
    panel_w = 280  # 加宽以容纳 GT + 预测对比
    out_h, out_w = h, w + panel_w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))

    for ti, frame in enumerate(frames_np):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = (40, 40, 48)

        # 绘制 YOLO 框（需指定 -m 且 --draw-boxes）
        if args.yolo_model and args.draw_boxes and ti < len(results_list) and results_list[ti].boxes is not None:
            res = results_list[ti]
            for i in range(len(res.boxes)):
                x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy()
                tid = int(res.boxes.id[i]) if res.boxes.id is not None else 0
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(frame, str(tid), (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0))

        canvas[:, :w] = frame

        # 右侧标注面板：GT（真值） vs Pred（LSTM 预测）
        y0 = 15
        cv2.putText(canvas, "GT vs LSTM", (w + 8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y0 += 22
        for si in sorted(labels_per_frame.get(ti, {}).keys()):
            gt_label, gt_reason, pred_label, pred_reason, match = labels_per_frame[ti][si]
            # 蛇编号
            cv2.putText(canvas, f"Snake {si+1}", (w + 8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            y0 += 18
            # GT（真值）
            gt_ok = gt_label == "correct"
            cv2.putText(canvas, "GT:", (w + 8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
            cv2.putText(canvas, ("OK" if gt_ok else "NG") + " " + (REASON_ZH.get(gt_reason, gt_reason)[:10]),
                        (w + 38, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (76, 175, 80) if gt_ok else (244, 67, 54), 1)
            y0 += 16
            # Pred（预测）
            pred_ok = (pred_label == "correct") if pred_label != "-" else None
            cv2.putText(canvas, "Pred:", (w + 8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
            pred_txt = (("OK" if pred_ok else "NG") + " " + REASON_ZH.get(pred_reason, pred_reason)[:10]) if pred_label != "-" else "-"
            pred_color = (76, 175, 80) if pred_ok else (244, 67, 54) if pred_ok is False else (150, 150, 150)
            cv2.putText(canvas, pred_txt, (w + 38, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pred_color, 1)
            # 一致性
            match_txt = " [OK]" if match else " [X]"
            match_color = (76, 175, 80) if match else (244, 67, 54)
            cv2.putText(canvas, match_txt, (w + 200, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.35, match_color, 1)
            y0 += 22

        cv2.putText(canvas, f"Frame {ti+1}/{len(frames_np)}", (w + 8, out_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        writer.write(canvas)

    writer.release()

    # 统计准确率
    n_total = len(snake_preds)
    n_match = 0
    n_label_match = 0  # 仅 label 一致
    for si in snake_preds:
        for ti in range(len(scenes)):
            if si in labels_per_frame.get(ti, {}):
                gt_l, gt_r, pred_l, pred_r, m = labels_per_frame[ti][si]
                n_match += 1 if m else 0
                n_label_match += 1 if (gt_l == pred_l and pred_l != "-") else 0
                break
    acc = n_match / n_total * 100 if n_total else 0
    label_acc = n_label_match / n_total * 100 if n_total else 0
    print(f"已输出: {args.output}")
    print(f"真值 vs LSTM: {n_match}/{n_total} 完全一致(label+reason), {n_label_match}/{n_total} label一致")
    print(f"  完全一致准确率 {acc:.1f}%, label 准确率 {label_acc:.1f}%")
    for si in sorted(snake_preds.keys()):
        for ti in range(len(scenes)):
            if si in labels_per_frame.get(ti, {}):
                gt_l, gt_r, pred_l, pred_r, _ = labels_per_frame[ti][si]
                print(f"  蛇{si+1}: GT={gt_l}/{gt_r}  Pred={pred_l}/{pred_r}")
                break


if __name__ == "__main__":
    main()

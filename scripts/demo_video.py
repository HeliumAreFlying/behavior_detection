"""
实战测试：将对局记录渲染为视频，YOLO+行为模型联合标注，输出带标注的视频。

用法:
  python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c checkpoints/behavior/best.pt -o demo.mp4
  python scripts/demo_video.py -b batches/batch_00000.json -e 0 -m yolov8n.pt -c checkpoints/behavior/best.pt -o demo.mp4 --draw-boxes
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GRID_W = GRID_H = 15
CELL_SIZE = 40
IMG_W = IMG_H = GRID_W * CELL_SIZE  # 600


def _scene_to_features(scene: dict) -> dict[int, dict]:
    """从 scene 提取每蛇的 (head_x, head_y, food_x, food_y, x2_x, x2_y, has_x2) 归一化"""
    out: dict[int, dict] = {}
    snakes = scene.get("snakes", [])
    for si, s in enumerate(snakes):
        body = s.get("body", [])
        food = s.get("food", [0, 0])
        x2 = s.get("x2")
        if not body:
            continue
        hx, hy = body[0][0] % GRID_W, body[0][1] % GRID_H
        xc = (hx + 0.5) / GRID_W
        yc = (hy + 0.5) / GRID_H
        fx = (int(food[0]) % GRID_W + 0.5) / GRID_W if food else 0.0
        fy = (int(food[1]) % GRID_H + 0.5) / GRID_H if food else 0.0
        xx = (int(x2[0]) % GRID_W + 0.5) / GRID_W if x2 else 0.0
        xy = (int(x2[1]) % GRID_H + 0.5) / GRID_H if x2 else 0.0
        has_x2 = 1.0 if x2 else 0.0
        out[si] = {"xc": xc, "yc": yc, "fx": fx, "fy": fy, "xx": xx, "xy": xy, "has_x2": has_x2}
    return out


def _build_seq_features(scene_features: list[dict[int, dict]]) -> dict[int, list[list[float]]]:
    """从 scene 构建每蛇的 12 维特征序列"""
    snake_seqs: dict[int, list[list[float]]] = {}
    num_snakes = max(len(sf) for sf in scene_features) if scene_features else 0
    for si in range(num_snakes):
        seq = []
        for t, sf in enumerate(scene_features):
            if si not in sf:
                continue
            f = sf[si]
            xc, yc = f["xc"], f["yc"]
            fx, fy = f["fx"], f["fy"]
            xx, xy = f["xx"], f["xy"]
            has_x2 = f["has_x2"]
            if t > 0 and si in scene_features[t - 1]:
                prev = scene_features[t - 1][si]
                dx = xc - prev["xc"]
                dy = yc - prev["yc"]
            else:
                dx, dy = 0.0, 0.0
            df = min(((fx - xc) ** 2 + (fy - yc) ** 2) ** 0.5, 1.5) if (fx or fy) else 0.0
            dx2 = min(((xx - xc) ** 2 + (xy - yc) ** 2) ** 0.5, 1.5) if has_x2 and (xx or xy) else 0.0
            vel = (dx * dx + dy * dy) ** 0.5 or 1e-6
            to_food = (fx - xc) * dx + (fy - yc) * dy
            move_to_food = max(-1, min(1, to_food / vel)) if (fx or fy) else 0.0
            seq.append([xc, yc, dx, dy, fx, fy, xx, xy, has_x2, df, dx2, move_to_food])
        if len(seq) >= 2:
            snake_seqs[si] = seq
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
    args = p.parse_args()

    try:
        import pygame
        import cv2
    except ImportError as e:
        print(f"请安装依赖: pip install pygame opencv-python\n{e}")
        sys.exit(1)

    from render_and_export import render_scene
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

    pygame.init()
    pygame.display.set_mode((1, 1))

    # 渲染帧
    frames_np = []
    scene_features = []
    for sc in scenes:
        surf = render_scene(sc)
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
    else:
        state, input_dim, hidden_dim, num_layers = ckpt, 12, 128, 2

    beh_model = BehaviorCorrectnessModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    beh_model.load_state_dict(state)
    beh_model.eval()

    # 用 scene 构建序列（基于游戏状态，不依赖 YOLO）
    snake_seqs = _build_seq_features(scene_features)

    # 对每条蛇跑行为模型，取每个端点的预测
    snake_preds: dict[int, list[tuple[int, str, str, float]]] = defaultdict(list)
    for si, seq in snake_seqs.items():
        with torch.no_grad():
            x = torch.tensor([seq], dtype=torch.float32)
            logits_c, logits_r, logits_ep = beh_model(x, None)
            pred_c = logits_c.argmax(1).item()
            pred_r = logits_r.argmax(1).item()
            ep_prob = torch.sigmoid(logits_ep).item()
        label = "incorrect" if pred_c == 1 else "correct"
        reason = REASON_NAMES[pred_r]
        snake_preds[si] = [(len(seq) - 1, label, reason, ep_prob)]

    # 每帧展示：仅对当前帧存在的蛇显示预测
    labels_per_frame: dict[int, dict[int, tuple[str, str]]] = defaultdict(dict)
    for ti in range(len(scenes)):
        for si in scene_features[ti]:
            if si in snake_preds:
                _, label, reason, _ = snake_preds[si][0]
                labels_per_frame[ti][si] = (label, reason)

    # 绘制并写视频
    h, w = frames_np[0].shape[:2]
    panel_w = 200
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

        # 右侧标注面板
        y0 = 20
        for si in sorted(labels_per_frame.get(ti, {}).keys()):
            label, reason = labels_per_frame[ti][si]
            color = (76, 175, 80) if label == "correct" else (244, 67, 54)
            txt = f"蛇{si+1}: {'正确' if label == 'correct' else '错误'}"
            cv2.putText(canvas, txt, (w + 10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y0 += 25
            reason_zh = REASON_ZH.get(reason, reason)
            cv2.putText(canvas, reason_zh[:18], (w + 10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y0 += 30

        cv2.putText(canvas, f"Frame {ti+1}/{len(frames_np)}", (w + 10, out_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        writer.write(canvas)

    writer.release()
    print(f"已输出: {args.output}")


if __name__ == "__main__":
    main()

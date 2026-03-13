"""
从 dataset 生成行为序列，供 LSTM 训练。两种模式:
  1. --from-labels: 直接从 label 文件读取蛇头坐标，无需 YOLO
  2. 默认: 运行 YOLO 跟踪，从检测结果提取序列（需 --model）

用法:
  python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences
  python scripts/run_track_and_prepare.py -m yolov8n.pt -d dataset -o sequences
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_HEAD, CLASS_FOOD, CLASS_X2 = 0, 2, 3


def _parse_label_per_snake(lbl_path: Path) -> list[tuple[float, float, float, float, float, float, float]]:
    """
    从 label 解析每蛇的 (head_x, head_y, food_x, food_y, x2_x, x2_y, has_x2)。
    顺序: snake_head(0), snake_body(1)..., food(2), x2(3)
    """
    if not lbl_path.exists():
        return []
    lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    snakes: list[tuple[float, float, float, float, float, float, float]] = []
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 5:
            i += 1
            continue
        cls_id = int(parts[0])
        xc, yc = float(parts[1]), float(parts[2])
        if cls_id == CLASS_HEAD:
            food_x, food_y = 0.0, 0.0
            x2_x, x2_y = 0.0, 0.0
            has_x2 = 0.0
            i += 1
            while i < len(lines):
                p = lines[i].split()
                if len(p) < 5:
                    i += 1
                    continue
                c = int(p[0])
                if c == CLASS_HEAD:
                    break
                if c == CLASS_FOOD:
                    food_x, food_y = float(p[1]), float(p[2])
                elif c == CLASS_X2:
                    x2_x, x2_y = float(p[1]), float(p[2])
                    has_x2 = 1.0
                i += 1
            snakes.append((xc, yc, food_x, food_y, x2_x, x2_y, has_x2))
        else:
            i += 1
    return snakes


def _gt_heads_from_labels(lbl_path: Path) -> list[tuple[float, float]]:
    """从 YOLO label 文件读取蛇头位置 (class=0)，返回 [(xc, yc), ...] 归一化坐标，按蛇顺序"""
    if not lbl_path.exists():
        return []
    gt: list[tuple[float, float]] = []
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(parts[0]) == CLASS_HEAD:
            gt.append((float(parts[1]), float(parts[2])))
    return gt


def match_tracks_to_snakes(
    head_detections: list[tuple[int, float, float]],  # [(track_id, xc, yc), ...]
    gt_heads: list[tuple[float, float]],  # [(xc, yc), ...] 归一化坐标
) -> dict[int, int]:
    """首帧蛇头检测 track_id -> 蛇下标 snake_idx。按位置最近匹配。"""
    if not head_detections or not gt_heads:
        return {}

    used_snake = set()
    track_to_snake: dict[int, int] = {}

    def dist(xc: float, yc: float, gxc: float, gyc: float) -> float:
        return (xc - gxc) ** 2 + (yc - gyc) ** 2

    sorted_dets = sorted(head_detections, key=lambda x: (x[1], x[2]))
    for track_id, xc, yc in sorted_dets:
        best_si, best_d = -1, float("inf")
        for si in range(len(gt_heads)):
            if si in used_snake:
                continue
            d = dist(xc, yc, gt_heads[si][0], gt_heads[si][1])
            if d < best_d:
                best_d, best_si = d, si
        if best_si >= 0:
            track_to_snake[track_id] = best_si
            used_snake.add(best_si)

    return track_to_snake


def extract_head_features_per_frame(
    results_list: list,
    track_to_snake: dict[int, int],
) -> dict[int, list[dict]]:
    """
    从每帧跟踪结果中提取每条蛇的 head 中心 (xc, yc)。
    返回: snake_idx -> [{"xc": f, "yc": f, "t": int}, ...]
    """
    snake_seqs: dict[int, list[dict]] = defaultdict(list)
    for t, res in enumerate(results_list):
        if res is None or res.boxes is None:
            continue
        boxes = res.boxes
        for i in range(len(boxes)):
            tid = int(boxes.id[i]) if boxes.id is not None else None
            cls_id = int(boxes.cls[i])
            if cls_id != CLASS_HEAD or tid is None:
                continue
            if tid not in track_to_snake:
                continue
            snake_idx = track_to_snake[tid]
            xc = float(boxes.xywhn[i][0])
            yc = float(boxes.xywhn[i][1])
            snake_seqs[snake_idx].append({"xc": xc, "yc": yc, "t": t})
    return dict(snake_seqs)


def extract_sequences_from_labels(
    entries: list[dict], dataset_dir: Path
) -> dict[int, list[dict]]:
    """
    从 label 文件读取每蛇每帧特征: head, food, x2。
    返回: snake_idx -> [{"xc", "yc", "fx", "fy", "xx", "xy", "has_x2", "t"}, ...]
    """
    snake_seqs: dict[int, list[dict]] = defaultdict(list)
    for t, e in enumerate(entries):
        lbl_path = dataset_dir / e["split"] / "labels" / f"{e['id']}.txt"
        rows = _parse_label_per_snake(lbl_path)
        for si, (hx, hy, fx, fy, xx, xy, has_x2) in enumerate(rows):
            snake_seqs[si].append({
                "xc": hx, "yc": hy,
                "fx": fx, "fy": fy, "xx": xx, "xy": xy, "has_x2": has_x2,
                "t": t,
            })
    return dict(snake_seqs)


def main():
    import argparse

    p = argparse.ArgumentParser(description="从 dataset 生成行为序列")
    p.add_argument("--from-labels", action="store_true",
                   help="直接从 label 读取蛇头坐标，无需 YOLO 模型")
    p.add_argument("--model", "-m", default="", help="YOLO 权重（非 from-labels 时必填）")
    p.add_argument("--dataset", "-d", default="dataset",
                   help="数据集根目录（含 train/val/labels/behavior, metadata.json）")
    p.add_argument("--output", "-o", default="sequences", help="输出目录")
    p.add_argument("--max-episodes", type=int, default=0, help="最多处理多少 episode，0 表示全部")
    p.add_argument("--device", default="", help="cuda:0 / cpu（仅 YOLO 模式）")
    args = p.parse_args()

    if not args.from_labels and not args.model:
        p.error("请指定 --from-labels 或 --model")

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute() and not dataset_dir.exists():
        dataset_dir = ROOT / dataset_dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        print(f"未找到 metadata.json，请先运行 render_and_export.py 生成数据集。")
        print(f"  python scripts/render_and_export.py -o {dataset_dir}")
        sys.exit(1)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    by_episode: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for m in metadata:
        key = (m["batch"], m["episode"])
        by_episode[key].append(m)
    for key in by_episode:
        by_episode[key].sort(key=lambda x: x["scene"])

    episode_keys = sorted(by_episode.keys())
    if args.max_episodes > 0:
        episode_keys = episode_keys[: args.max_episodes]

    if not args.from_labels:
        from ultralytics import YOLO
        model = YOLO(args.model)
        device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")

    all_sequences: list[dict] = []
    for ki, (batch_name, ep_idx) in enumerate(episode_keys):
        entries = by_episode[(batch_name, ep_idx)]
        if not entries:
            continue

        if args.from_labels:
            snake_seqs = extract_sequences_from_labels(entries, dataset_dir)
        else:
            img_paths = [dataset_dir / e["split"] / "images" / f"{e['id']}.png" for e in entries]
            missing = [p for p in img_paths if not p.exists()]
            if missing:
                print(f"[WARN] 跳过 {batch_name} ep{ep_idx}: 缺少 {len(missing)} 张图")
                continue
            results = model.track(
                source=[str(p) for p in img_paths],
                persist=True,
                tracker="bytetrack.yaml",
                device=device,
                verbose=False,
                stream=True,
            )
            results_list = list(results)
            head_dets: list[tuple[int, float, float]] = []
            match_frame_idx = -1
            for t, res in enumerate(results_list):
                if res is None or res.boxes is None or res.boxes.id is None:
                    continue
                for i in range(len(res.boxes)):
                    if int(res.boxes.cls[i]) != CLASS_HEAD:
                        continue
                    tid = int(res.boxes.id[i])
                    xc, yc = float(res.boxes.xywhn[i][0]), float(res.boxes.xywhn[i][1])
                    head_dets.append((tid, xc, yc))
                if head_dets:
                    match_frame_idx = t
                    break
            if match_frame_idx < 0 or not head_dets:
                continue
            match_entry = entries[match_frame_idx]
            lbl_path = dataset_dir / match_entry["split"] / "labels" / f"{match_entry['id']}.txt"
            gt_heads = _gt_heads_from_labels(lbl_path)
            if not gt_heads:
                continue
            track_to_snake = match_tracks_to_snakes(head_dets, gt_heads)
            yolo_seqs = extract_head_features_per_frame(results_list, track_to_snake)
            label_seqs = extract_sequences_from_labels(entries, dataset_dir)
            snake_seqs = {}
            for si in label_seqs:
                yolo_by_t = {f["t"]: f for f in yolo_seqs.get(si, [])}
                merged = []
                for lbl in label_seqs[si]:
                    t = lbl["t"]
                    yolo_f = yolo_by_t.get(t)
                    if yolo_f:
                        merged.append({**lbl, "xc": yolo_f["xc"], "yc": yolo_f["yc"]})
                    else:
                        merged.append(lbl)
                snake_seqs[si] = merged

        # 每帧的 snake_annotations 来自 behavior JSON；生成多样本供端点检测学习
        for snake_idx, seq in snake_seqs.items():
            if not seq:
                continue
            labels_per_frame: list[dict] = []
            for e in entries:
                beh_path = dataset_dir / e["split"] / "behavior" / f"{e['id']}.json"
                if beh_path.exists():
                    beh = json.loads(beh_path.read_text(encoding="utf-8"))
                    anns = beh.get("snake_annotations", [])
                    if snake_idx < len(anns):
                        labels_per_frame.append(anns[snake_idx])
                    else:
                        labels_per_frame.append({"label": "correct", "reason": "in_progress"})
                else:
                    labels_per_frame.append({"label": "correct", "reason": "in_progress"})

            n_frames = len(entries)
            endpoint_frames: list[int] = [
                i for i in range(n_frames)
                if i < len(labels_per_frame) and labels_per_frame[i].get("reason") != "in_progress"
            ]
            non_endpoint_frames: list[int] = [
                i for i in range(n_frames)
                if i not in set(endpoint_frames)
            ]

            # 端点样本：每帧 reason!=in_progress 一条
            for i in endpoint_frames:
                prefix = [f for f in seq if f["t"] <= i]
                if len(prefix) < 2:
                    continue
                ann = labels_per_frame[i]
                all_sequences.append({
                    "batch": batch_name,
                    "episode": ep_idx,
                    "snake_idx": snake_idx,
                    "features": prefix,
                    "label": ann["label"],
                    "reason": ann["reason"],
                    "is_endpoint": 1,
                })

            # 非端点样本：每「波」抽 1 个 in_progress 帧，让模型学「暂不输出」
            last_end = -1
            for i in sorted(endpoint_frames):
                mid = (last_end + 1 + i) // 2
                if mid > last_end and mid < i and mid in non_endpoint_frames:
                    prefix = [f for f in seq if f["t"] <= mid]
                    if len(prefix) >= 2:
                        all_sequences.append({
                            "batch": batch_name,
                            "episode": ep_idx,
                            "snake_idx": snake_idx,
                            "features": prefix,
                            "label": "correct",
                            "reason": "in_progress",
                            "is_endpoint": 0,
                        })
                last_end = i
            if endpoint_frames:
                mid = (last_end + 1 + n_frames) // 2
                if mid > last_end and mid < n_frames and mid in non_endpoint_frames:
                    prefix = [f for f in seq if f["t"] <= mid]
                    if len(prefix) >= 2:
                        all_sequences.append({
                            "batch": batch_name,
                            "episode": ep_idx,
                            "snake_idx": snake_idx,
                            "features": prefix,
                            "label": "correct",
                            "reason": "in_progress",
                            "is_endpoint": 0,
                        })

        if (ki + 1) % 50 == 0 or ki == len(episode_keys) - 1:
            print(f"已处理 {ki + 1}/{len(episode_keys)} episodes, 共 {len(all_sequences)} 条蛇序列")

    out_path = output_dir / "track_sequences.json"
    out_path.write_text(json.dumps(all_sequences, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: 输出 {out_path}，共 {len(all_sequences)} 条蛇序列")
    print("可用于 LSTM 行为/正确性模型训练。")


if __name__ == "__main__":
    main()

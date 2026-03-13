"""
在渲染图像序列上运行 YOLO 检测 + 跟踪，按 episode 分组，输出每蛇时序特征与行为标签。
仅依赖 dataset（含 metadata.json、train/val 的 images/labels/behavior），无需 batches。

依赖: pip install ultralytics
用法:
  python scripts/run_track_and_prepare.py --model runs/detect/train/weights/best.pt --dataset dataset --output sequences
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_HEAD = 0


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


def main():
    import argparse

    p = argparse.ArgumentParser(description="YOLO 跟踪 + 序列准备")
    p.add_argument("--model", "-m", required=True, help="YOLO 权重路径，如 runs/detect/train/weights/best.pt")
    p.add_argument("--dataset", "-d", default="dataset",
                   help="数据集根目录（含 train/val/images, metadata.json），可写绝对或相对路径")
    p.add_argument("--output", "-o", default="sequences", help="输出目录")
    p.add_argument("--max-episodes", type=int, default=0, help="最多处理多少 episode，0 表示全部")
    p.add_argument("--device", default="", help="cuda:0 / cpu，默认自动")
    args = p.parse_args()

    from ultralytics import YOLO

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

    model = YOLO(args.model)
    device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")

    episode_keys = sorted(by_episode.keys())
    if args.max_episodes > 0:
        episode_keys = episode_keys[: args.max_episodes]

    all_sequences: list[dict] = []
    for ki, (batch_name, ep_idx) in enumerate(episode_keys):
        entries = by_episode[(batch_name, ep_idx)]
        img_paths: list[Path] = []
        for e in entries:
            split = e["split"]
            img_paths.append(dataset_dir / split / "images" / f"{e['id']}.png")
        if not img_paths:
            continue

        # 检查文件存在
        missing = [p for p in img_paths if not p.exists()]
        if missing:
            print(f"[WARN] 跳过 {batch_name} ep{ep_idx}: 缺少 {len(missing)} 张图")
            continue

        # YOLO track，persist=True 保持跨帧 track_id
        results = model.track(
            source=[str(p) for p in img_paths],
            persist=True,
            device=device,
            verbose=False,
            stream=True,
        )
        results_list = list(results)

        # 首帧 label 文件提供 GT 蛇头位置，用于 track -> snake 匹配
        first_entry = entries[0]
        lbl_path = dataset_dir / first_entry["split"] / "labels" / f"{first_entry['id']}.txt"
        gt_heads = _gt_heads_from_labels(lbl_path)
        if not gt_heads:
            continue

        # 首帧蛇头检测
        head_dets: list[tuple[int, float, float]] = []
        r0 = results_list[0] if results_list else None
        if r0 and r0.boxes is not None:
            for i in range(len(r0.boxes)):
                if int(r0.boxes.cls[i]) != CLASS_HEAD:
                    continue
                tid = int(r0.boxes.id[i]) if r0.boxes.id is not None else None
                if tid is None:
                    continue
                xc, yc = float(r0.boxes.xywhn[i][0]), float(r0.boxes.xywhn[i][1])
                head_dets.append((tid, xc, yc))

        track_to_snake = match_tracks_to_snakes(head_dets, gt_heads)
        snake_seqs = extract_head_features_per_frame(results_list, track_to_snake)

        # 每帧的 snake_annotations 来自 behavior JSON
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

            # 取最后一帧的 label/reason 作为整条蛇序列的标注（针对当前食物）
            last_label = labels_per_frame[-1] if labels_per_frame else {"label": "correct", "reason": "in_progress"}

            rec = {
                "batch": batch_name,
                "episode": ep_idx,
                "snake_idx": snake_idx,
                "features": seq,
                "labels_per_frame": labels_per_frame,
                "label": last_label["label"],
                "reason": last_label["reason"],
            }
            all_sequences.append(rec)

        if (ki + 1) % 50 == 0 or ki == len(episode_keys) - 1:
            print(f"已处理 {ki + 1}/{len(episode_keys)} episodes, 共 {len(all_sequences)} 条蛇序列")

    out_path = output_dir / "track_sequences.json"
    out_path.write_text(json.dumps(all_sequences, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: 输出 {out_path}，共 {len(all_sequences)} 条蛇序列")
    print("可用于 LSTM 行为/正确性模型训练。")


if __name__ == "__main__":
    main()

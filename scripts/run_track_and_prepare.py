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
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_HEAD, CLASS_FOOD, CLASS_X2, CLASS_HEAD_DEAD = 0, 2, 3, 4


def _parse_label_per_snake(lbl_path: Path) -> list[tuple[float, float, float, float, float, float, float, float]]:
    """
    从 label 解析每蛇的 (head_x, head_y, food_x, food_y, x2_x, x2_y, has_x2, is_dead)。
    snake_head(0) / snake_head_dead(4) 区分活/死
    """
    if not lbl_path.exists():
        return []
    lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    snakes: list[tuple[float, float, float, float, float, float, float, float]] = []
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 5:
            i += 1
            continue
        cls_id = int(parts[0])
        xc, yc = float(parts[1]), float(parts[2])
        if cls_id in (CLASS_HEAD, CLASS_HEAD_DEAD):
            is_dead = 1.0 if cls_id == CLASS_HEAD_DEAD else 0.0
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
                if c in (CLASS_HEAD, CLASS_HEAD_DEAD):
                    break
                if c == CLASS_FOOD:
                    food_x, food_y = float(p[1]), float(p[2])
                elif c == CLASS_X2:
                    x2_x, x2_y = float(p[1]), float(p[2])
                    has_x2 = 1.0
                i += 1
            snakes.append((xc, yc, food_x, food_y, x2_x, x2_y, has_x2, is_dead))
        else:
            i += 1
    return snakes


def _gt_heads_from_labels(lbl_path: Path) -> list[tuple[float, float]]:
    """从 YOLO label 文件读取蛇头位置 (class=0 或 4)，返回 [(xc, yc), ...] 归一化坐标，按蛇顺序"""
    if not lbl_path.exists():
        return []
    gt: list[tuple[float, float]] = []
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) >= 5 and int(parts[0]) in (CLASS_HEAD, CLASS_HEAD_DEAD):
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
    从每帧跟踪结果中提取每条蛇的 head (xc, yc, is_dead)。
    YOLO 需 5 类时 class 4=snake_head_dead 表示撞击死亡。
    """
    snake_seqs: dict[int, list[dict]] = defaultdict(list)
    for t, res in enumerate(results_list):
        if res is None or res.boxes is None:
            continue
        boxes = res.boxes
        for i in range(len(boxes)):
            tid = int(boxes.id[i]) if boxes.id is not None else None
            cls_id = int(boxes.cls[i])
            if cls_id not in (CLASS_HEAD, CLASS_HEAD_DEAD) or tid is None:
                continue
            if tid not in track_to_snake:
                continue
            snake_idx = track_to_snake[tid]
            xc = float(boxes.xywhn[i][0])
            yc = float(boxes.xywhn[i][1])
            is_dead = 1.0 if cls_id == CLASS_HEAD_DEAD else 0.0
            snake_seqs[snake_idx].append({"xc": xc, "yc": yc, "is_dead": is_dead, "t": t})
    return dict(snake_seqs)


def _infer_ate_events(
    prev: tuple | None,
    curr: tuple,
) -> tuple[float, float]:
    """根据前后帧推断本帧是否刚吃到食物/x2。返回 (ate_food, ate_x2)。"""
    if prev is None:
        return 0.0, 0.0
    hx, hy, fx, fy, xx, xy, has_x2 = curr[:7]
    px, py, pfx, pfy, pxx, pxy, phx2 = prev[:7]
    thresh = 0.02  # 归一化坐标下距离阈值
    def _dist(ax, ay, bx, by) -> float:
        return (ax - bx) ** 2 + (ay - by) ** 2
    ate_food = 0.0
    if (pfx or pfy) and _dist(fx, fy, pfx, pfy) > thresh:  # 食物位置变化
        if _dist(hx, hy, pfx, pfy) < thresh:  # 蛇头在上一帧食物位置
            ate_food = 1.0
    ate_x2 = 0.0
    if phx2 and (not has_x2 or _dist(xx, xy, pxx, pxy) > thresh):  # x2 消失或位移
        if (pxx or pxy) and _dist(hx, hy, pxx, pxy) < thresh:  # 蛇头在上一帧 x2 位置
            ate_x2 = 1.0
    return ate_food, ate_x2


def extract_sequences_from_labels(
    entries: list[dict], dataset_dir: Path
) -> dict[int, list[dict]]:
    """
    从 label + behavior 文件读取每蛇每帧特征。
    含 head_forward_type(0/1/2/3: 空/己身/他蛇身体/他蛇头), ate_food_while_x2_exists。
    """
    snake_seqs: dict[int, list[dict]] = defaultdict(list)
    prev_rows: list[tuple] | None = None
    steps_counters: list[int] = []
    for t, e in enumerate(entries):
        lbl_path = dataset_dir / e["split"] / "labels" / f"{e['id']}.txt"
        beh_path = dataset_dir / e["split"] / "behavior" / f"{e['id']}.json"
        rows = _parse_label_per_snake(lbl_path)
        head_forward_list: list[int] = []
        if beh_path.exists():
            beh = json.loads(beh_path.read_text(encoding="utf-8"))
            head_forward_list = beh.get("head_forward_type", [])
        if t == 0:
            steps_counters = [0] * len(rows)
        while len(steps_counters) < len(rows):
            steps_counters.append(0)
        for si, curr in enumerate(rows):
            hx, hy, fx, fy, xx, xy, has_x2, is_dead = curr[:8]
            prev = prev_rows[si] if prev_rows and si < len(prev_rows) else None
            ate_food, ate_x2 = _infer_ate_events(prev, curr)
            ate_food_while_x2 = 1.0 if (ate_food and has_x2) else 0.0
            head_forward = int(head_forward_list[si]) if si < len(head_forward_list) else 0
            head_forward = max(0, min(3, head_forward))
            if ate_food:
                steps_counters[si] = 0
            else:
                steps_counters[si] = steps_counters[si] + 1
            steps_since_food = min(steps_counters[si] / 80.0, 1.0)
            # 下一步再没吃到果子就超时(80步)：当前已 79 步未吃则为 1
            about_to_timeout = 1.0 if steps_counters[si] >= 79 else 0.0
            snake_seqs[si].append({
                "xc": hx, "yc": hy,
                "fx": fx, "fy": fy, "xx": xx, "xy": xy, "has_x2": has_x2,
                "ate_food": ate_food, "ate_x2": ate_x2,
                "ate_food_while_x2_exists": ate_food_while_x2,
                "head_forward_type": head_forward,
                "is_dead": is_dead, "steps_since_food": steps_since_food,
                "about_to_timeout": about_to_timeout,
                "t": t,
            })
        prev_rows = rows
    return dict(snake_seqs)


_worker_model = None
_worker_device = None


def _pool_join_timeout(pool, timeout: float = 3.0) -> None:
    """等待 Pool 子进程退出，每个进程最多等待 timeout 秒，避免 join() 无限阻塞"""
    import time
    if not hasattr(pool, "_pool"):
        return
    t0 = time.perf_counter()
    for p in pool._pool:
        remain = max(0, timeout - (time.perf_counter() - t0))
        if remain > 0 and p.is_alive():
            p.join(timeout=remain)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)


def _init_yolo_worker(model_path: str, device: str) -> None:
    """子进程内加载 YOLO 模型"""
    global _worker_model, _worker_device
    from ultralytics import YOLO
    _worker_model = YOLO(model_path)
    _worker_device = device


def _process_one_episode(args: tuple) -> tuple[int, list[dict]]:
    """
    处理单个 episode，返回 (ki, sequences)。
    args: (ki, batch_name, ep_idx, entries, dataset_dir_str, from_labels, model_path, device)
    """
    ki, batch_name, ep_idx, entries, dataset_dir_str, from_labels, model_path, device = args
    dataset_dir = Path(dataset_dir_str)
    sequences: list[dict] = []

    if not entries:
        return ki, []

    # 与 render_and_export 的 train/val 一致：按该 episode 内帧的 split 众数作为整条序列的 split
    splits_in_ep = [e.get("split", "train") for e in entries]
    episode_split = Counter(splits_in_ep).most_common(1)[0][0] if splits_in_ep else "train"

    if from_labels:
        snake_seqs = extract_sequences_from_labels(entries, dataset_dir)
    else:
        img_paths = [dataset_dir / e["split"] / "images" / f"{e['id']}.png" for e in entries]
        missing = [p for p in img_paths if not p.exists()]
        if missing:
            return ki, []  # 主进程会看到空，不打印 warn 减少噪音
        results = _worker_model.track(
            source=[str(p) for p in img_paths],
            persist=True,
            tracker="bytetrack.yaml",
            device=_worker_device,
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
            return ki, []
        match_entry = entries[match_frame_idx]
        lbl_path = dataset_dir / match_entry["split"] / "labels" / f"{match_entry['id']}.txt"
        gt_heads = _gt_heads_from_labels(lbl_path)
        if not gt_heads:
            return ki, []
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
                    merged.append({
                        **lbl, "xc": yolo_f["xc"], "yc": yolo_f["yc"],
                        "is_dead": yolo_f.get("is_dead", lbl["is_dead"]),
                    })
                else:
                    merged.append(lbl)
            snake_seqs[si] = merged

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
        non_endpoint_frames: list[int] = [i for i in range(n_frames) if i not in set(endpoint_frames)]

        for i in endpoint_frames:
            prefix = [f for f in seq if f["t"] <= i]
            if len(prefix) < 2:
                continue
            ann = labels_per_frame[i]
            sequences.append({
                "batch": batch_name,
                "episode": ep_idx,
                "snake_idx": snake_idx,
                "features": prefix,
                "label": ann["label"],
                "reason": ann["reason"],
                "is_endpoint": 1,
                "split": episode_split,
            })

        last_end = -1
        for i in sorted(endpoint_frames):
            mid = (last_end + 1 + i) // 2
            if mid > last_end and mid < i and mid in non_endpoint_frames:
                prefix = [f for f in seq if f["t"] <= mid]
                if len(prefix) >= 2:
                    sequences.append({
                        "batch": batch_name,
                        "episode": ep_idx,
                        "snake_idx": snake_idx,
                        "features": prefix,
                        "label": "correct",
                        "reason": "in_progress",
                        "is_endpoint": 0,
                        "split": episode_split,
                    })
            last_end = i
        if endpoint_frames:
            mid = (last_end + 1 + n_frames) // 2
            if mid > last_end and mid < n_frames and mid in non_endpoint_frames:
                prefix = [f for f in seq if f["t"] <= mid]
                if len(prefix) >= 2:
                    sequences.append({
                        "batch": batch_name,
                        "episode": ep_idx,
                        "snake_idx": snake_idx,
                        "features": prefix,
                        "label": "correct",
                        "reason": "in_progress",
                        "is_endpoint": 0,
                        "split": episode_split,
                    })

    return ki, sequences


def main():
    import argparse
    import multiprocessing as mp

    p = argparse.ArgumentParser(description="从 dataset 生成行为序列")
    p.add_argument("--from-labels", action="store_true",
                   help="直接从 label 读取蛇头坐标，无需 YOLO 模型")
    p.add_argument("--model", "-m", default="", help="YOLO 权重（非 from-labels 时必填）")
    p.add_argument("--dataset", "-d", default="dataset",
                   help="数据集根目录（含 train/val/labels/behavior, metadata.json）")
    p.add_argument("--output", "-o", default="sequences", help="输出目录")
    p.add_argument("--workers", "-w", type=int, default=None,
                   help="并行进程数，默认 CPU 核心数（YOLO+GPU 时建议 1）")
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

    device = ""
    if not args.from_labels:
        device = args.device or ("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
        # YOLO+GPU 时默认单进程，避免显存争用
        workers = args.workers if args.workers is not None else (1 if "cuda" in device else (mp.cpu_count() or 4))
    else:
        workers = args.workers or (mp.cpu_count() or 4)

    workers = max(1, min(workers, len(episode_keys)))

    dataset_dir_str = str(dataset_dir.resolve())
    work_items = [
        (ki, batch_name, ep_idx, by_episode[(batch_name, ep_idx)],
         dataset_dir_str, args.from_labels, args.model, device)
        for ki, (batch_name, ep_idx) in enumerate(episode_keys)
    ]

    all_sequences: list[dict] = []
    if workers <= 1:
        if not args.from_labels:
            from ultralytics import YOLO
            globals()["_worker_model"] = YOLO(args.model)
            globals()["_worker_device"] = device
        for ki, (batch_name, ep_idx) in enumerate(episode_keys):
            entries = by_episode[(batch_name, ep_idx)]
            _, ep_seqs = _process_one_episode((ki, batch_name, ep_idx, entries, dataset_dir_str,
                                              args.from_labels, args.model, device))
            all_sequences.extend(ep_seqs)
            if (ki + 1) % 50 == 0 or ki == len(episode_keys) - 1:
                print(f"已处理 {ki + 1}/{len(episode_keys)} episodes, 共 {len(all_sequences)} 条蛇序列")
    else:
        print(f"使用 {workers} 个进程并行处理...")
        pool = (
            mp.Pool(processes=workers, initializer=_init_yolo_worker, initargs=(args.model, device))
            if not args.from_labels
            else mp.Pool(processes=workers)
        )
        interrupted = False
        try:
            processed = 0
            for ki, ep_seqs in pool.imap_unordered(_process_one_episode, work_items, chunksize=1):
                all_sequences.extend(ep_seqs)
                processed += 1
                if processed % 50 == 0 or processed == len(episode_keys):
                    print(f"已处理 {processed}/{len(episode_keys)} episodes, 共 {len(all_sequences)} 条蛇序列", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[中断] 用户取消，正在终止工作进程...", flush=True)
            pool.terminate()
            _pool_join_timeout(pool, timeout=3.0)
            raise SystemExit(130)
        finally:
            if not interrupted:
                pool.close()
                pool.join()

    out_path = output_dir / "track_sequences.json"
    out_path.write_text(json.dumps(all_sequences, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成: 输出 {out_path}，共 {len(all_sequences)} 条蛇序列")
    print("可用于 LSTM 行为/正确性模型训练。")


if __name__ == "__main__":
    main()

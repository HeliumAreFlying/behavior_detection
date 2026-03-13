"""
验证数据链路一致性：对比 demo 从 batch 提取的特征 vs run_track_and_prepare 从 dataset 提取的特征。

用法:
  python scripts/verify_pipeline.py -b batches/batch_00000.json -e 0 -d dataset
  # 需先运行 render_and_export 生成 dataset
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_track_and_prepare import extract_sequences_from_labels


def main():
    import argparse
    p = argparse.ArgumentParser(description="验证 demo 与 dataset 特征一致性")
    p.add_argument("--batch", "-b", required=True, help="batch JSON")
    p.add_argument("--episode", "-e", type=int, default=0)
    p.add_argument("--dataset", "-d", default="dataset", help="render_and_export 输出的 dataset 目录")
    args = p.parse_args()

    batch_path = Path(args.batch)
    if not batch_path.is_absolute():
        batch_path = ROOT / batch_path
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir

    if not batch_path.exists():
        print(f"batch 不存在: {batch_path}")
        sys.exit(1)
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        print(f"dataset 不存在或未运行 render_and_export: {meta_path}")
        sys.exit(1)

    data = json.loads(batch_path.read_text(encoding="utf-8"))
    batch_name = batch_path.name
    ep = data["episodes"][args.episode]
    scenes = ep.get("scenes", [])
    ep_reason = ep.get("reason", "")

    # 从 metadata 获取该 episode 的 entries
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    entries = [m for m in metadata if m.get("batch") == batch_name and m.get("episode") == args.episode]
    entries.sort(key=lambda x: x["scene"])

    if not entries:
        print(f"metadata 中无 {batch_name} episode {args.episode}")
        sys.exit(1)

    # 从 dataset 提取的特征（与 run_track_and_prepare 一致）
    label_seqs = extract_sequences_from_labels(entries, dataset_dir)

    # render_and_export 不跳帧，导出全帧
    expected_scene_indices = list(range(len(scenes)))

    print(f"batch={batch_name} episode={args.episode}")
    print(f"  scenes 总数: {len(scenes)}")
    print(f"  dataset entries 数: {len(entries)} (应等于 render 导出帧数)")

    if len(entries) != len(expected_scene_indices):
        print(f"  [WARN] 不一致! entries={len(entries)} vs 期望={len(expected_scene_indices)}")
        print("  可能原因: render_and_export 与 metadata 帧数不一致")
    else:
        print("  [OK] 帧数一致")

    # 对比 entries 的 scene 索引与期望
    entry_scenes = [e["scene"] for e in entries]
    if entry_scenes != expected_scene_indices:
        print(f"  [WARN] scene 索引不完全一致")
        print(f"    entries: {entry_scenes[:10]}... (共{len(entry_scenes)})")
        print(f"    期望:   {expected_scene_indices[:10]}... (共{len(expected_scene_indices)})")
    else:
        print("  [OK] scene 索引完全一致")

    # 对比每蛇的特征（dataset label vs demo 从 batch 构建）
    from demo_video import _scene_to_features, _build_seq_features

    filtered_sf = []
    for sc_idx in expected_scene_indices:
        if sc_idx >= len(scenes):
            break
        filtered_sf.append(_scene_to_features(scenes[sc_idx]))
    demo_seqs = _build_seq_features(filtered_sf, input_dim=14)

    for si in sorted(set(label_seqs.keys()) | set(demo_seqs.keys())):
        label_seq = label_seqs.get(si, [])
        demo_seq = demo_seqs.get(si, [])
        n_label = len(label_seq)
        n_demo = len(demo_seq)
        if n_label != n_demo:
            print(f"  蛇{si}: 长度不一致 dataset={n_label} demo={n_demo}")
            continue
        # 比 head (xc, yc)
        diff_count = 0
        for t in range(n_label):
            lb = label_seq[t]
            dm = demo_seq[t]
            if abs(lb["xc"] - dm[0]) > 1e-5 or abs(lb["yc"] - dm[1]) > 1e-5:
                diff_count += 1
        if diff_count > 0:
            print(f"  蛇{si}: head 坐标 {diff_count}/{n_demo} 帧有差异")
        else:
            print(f"  蛇{si}: [OK] 特征与 dataset 一致 ({n_demo} 帧)")


if __name__ == "__main__":
    main()

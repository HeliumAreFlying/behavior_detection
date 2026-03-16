# 命令速查（分行）

## 多进程与错误检测参数速查

| 脚本 | 多进程 | 错误检测相关 |
|------|--------|--------------|
| `data_generator.py` | `-w 12` | - |
| `render_and_export.py` | `-w 12` | - |
| `run_track_and_prepare.py` | `-w 12`（from-labels 时） | - |
| `train_behavior.py` | `--patience 50` 早停 | `--best-metric reason_f1` 稳健选 best；`--boost-incorrect`；`--no-bidirectional --no-attention` |
| `eval_behavior.py` | `--batch-size 256` 大批量推理提速 | 默认仅评估 **val** 集（`--split val`）；`--split all` 评估全部；`--no-threshold-search` 固定阈值；`--reason-override` |
| `infer_behavior.py` | - | `--incorrect-threshold 0.3` |

> Ctrl+C 可安全中断多进程脚本，不会卡死。

---

## 完整训练流程（含 YOLO）

渲染 640×640；蛇头：开局菱形 → 三角(尖指) → 撞击圆形。行为模型：双向 LSTM + 注意力。

```bash
# 1. 数据生成（-w 12 多进程）
python data_generator.py --batches 10 --batch-size 100 -w 12

# 2. 渲染与导出（640×640 全帧，-w 12 多进程）
python scripts/render_and_export.py --batches batches/ --output dataset --val-ratio 0.2 -w 12

# 3. (可选) YOLO 训练（imgsz=640 与渲染一致）
yolo train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=128

# 4. 序列准备
# 路径 A：纯 label（-w 12）
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences -w 12

# 路径 B：YOLO 跟踪（需步骤 3）
python scripts/run_track_and_prepare.py -m runs/detect/train/weights/best.pt -d dataset -o sequences

# 5. 行为模型训练（--best-metric reason_f1 稳健选 best；--patience 50 早停）
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior \
  --boost-incorrect --aug-multiscale --aug-frame-drop 0.1 --aug-noise 0.02 --epochs 1000 --patience 50

# 6. 评估（默认仅 val 集，与训练一致；--split all 评估全部；--batch-size 256 提速）
python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json --batch-size 256

# 7. 演示视频（-d dataset 与训练一致；路径 B 加 -m 权重 --draw-boxes）
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c checkpoints/behavior/best.pt -d dataset -o demo.mp4
```

---

## 1. 数据生成（支持多进程 `-w`）

```bash
python data_generator.py

# 多进程加速
python data_generator.py -b 100 -s 100 -w 12
```

## 2. 渲染与导出（640×640，支持多进程 `-w`）

划分按**局（episode）**：`--val-ratio 0.2` 表示约 20% 的局整局进入 val，避免按帧划分导致 val 样本过少。

```bash
python scripts/render_and_export.py --batches batches/ --output dataset --val-ratio 0.2

# 多进程加速
python scripts/render_and_export.py --batches batches/ -o dataset -w 12
```

## 3. 预览标注

```bash
python scripts/preview_labels.py --num 100
```

## 4. YOLO 训练

```bash
yolo train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=128
```

## 5. 序列准备（二选一，支持多进程 `-w`）

生成的 `track_sequences.json` 中每条序列带 `split`（train/val），与 `render_and_export` 按局划分的 dataset 一致；每帧含 `head_forward_type`（0=空/1=己身/2=他蛇身体/3=他蛇头），供训练与评估使用。

```bash
# 纯 label（多进程）
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences -w 12
```

```bash
# YOLO 跟踪（GPU 时默认单进程）
python scripts/run_track_and_prepare.py -m runs/detect/train/weights/best.pt -d dataset -o sequences
```

## 6. 行为模型训练（纯网格）

```bash
python scripts/train_behavior.py --data grid --batches batches/ -o checkpoints/behavior --epochs 100 --patience 50
```

## 7. 行为模型训练（序列数据，默认双向 LSTM + 注意力）

```bash
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences
```

```bash
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior
```

**推荐：启用类别平衡与增强；best 按 reason_f1 选取**

```bash
# 错误检测率低时使用 --boost-incorrect；--patience 50 早停
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior \
  --boost-incorrect --aug-multiscale --aug-frame-drop 0.1 --aug-noise 0.02 --patience 50

# --best-metric：reason_f1(默认) / binary_f1 / composite
python scripts/train_behavior.py -d sequences/track_sequences.json -o checkpoints/behavior --best-metric composite

# 禁用双向 LSTM / 自注意力（兼容旧 checkpoint）
python scripts/train_behavior.py -d sequences/track_sequences.json -o checkpoints/behavior --no-bidirectional --no-attention
```

## 8. 行为模型推理

```bash
python scripts/infer_behavior.py --model checkpoints/behavior/best.pt --data sequences/track_sequences.json

# --incorrect-threshold 0.3 可提升错误召回率
python scripts/infer_behavior.py -m best.pt -d sequences/track_sequences.json --incorrect-threshold 0.3
```

```bash
python scripts/infer_behavior.py -m checkpoints/behavior/best.pt -d grid --batches batches/
```

## 9. 实战演示视频（YOLO + 行为联合标注）

```bash
# 推荐：指定 dataset，从 metadata 读取实际导出帧，与训练 100% 一致
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c best.pt -d dataset -o demo.mp4

# 无 dataset 时用全帧（与 render 一致，不跳帧）
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c best.pt -o demo.mp4

# YOLO 框 + 行为标注
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -m yolov8n.pt -c best.pt -d dataset -o demo.mp4 --draw-boxes
```

## 10. 链路验证（确保 demo 与训练数据一致）

```bash
# 需先 render_and_export 生成 dataset
python scripts/verify_pipeline.py -b batches/batch_00000.json -e 0 -d dataset
```

## 11. 全量评估（P、R、mAP50、mAP50-95，YOLO 风格表格）

默认**仅评估验证集**（`--split val`），与训练时的 val 一致；旧版无 `split` 的 JSON 请用 `--split all`。

```bash
# 默认评估 val 集（与训练验证集一致）
python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json

# 评估全部：--split all
python scripts/eval_behavior.py -c best.pt -d sequences/track_sequences.json --split all

# 使用固定阈值
python scripts/eval_behavior.py -c best.pt -d sequences/track_sequences.json --no-threshold-search --incorrect-threshold 0.9

# --reason-override：reason 为错误类时强制 label=incorrect
python scripts/eval_behavior.py -c best.pt -d sequences/track_sequences.json --reason-override

# 大批量推理提速（默认 128，显存允许可增大）
python scripts/eval_behavior.py -c best.pt -d sequences/track_sequences.json --batch-size 256
```

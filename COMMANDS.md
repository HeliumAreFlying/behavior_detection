# 命令速查（分行）

## 多进程与错误检测参数速查

| 脚本 | 多进程 | 错误检测相关 |
|------|--------|--------------|
| `data_generator.py` | `-w 8` | - |
| `render_and_export.py` | `-w 8` | - |
| `run_track_and_prepare.py` | `-w 8`（from-labels 时） | - |
| `train_behavior.py` | - | `--boost-incorrect` 一键提升错误检测率；`--no-bidirectional --no-attention` 禁用新结构 |
| `eval_behavior.py` | - | 默认自动搜索最优 F1 阈值；`--no-threshold-search` 使用固定阈值；`--reason-override` |
| `infer_behavior.py` | - | `--incorrect-threshold 0.3` |

> Ctrl+C 可安全中断多进程脚本，不会卡死。

---

## 完整训练流程（含 YOLO）

渲染 640×640；蛇头：开局菱形 → 三角(尖指) → 撞击圆形。行为模型：双向 LSTM + 注意力。

```bash
# 1. 数据生成（-w 8 多进程）
python data_generator.py --batches 10 --batch-size 100 -w 8

# 2. 渲染与导出（640×640 全帧，-w 8 多进程）
python scripts/render_and_export.py --batches batches/ --output dataset --val-ratio 0.2 -w 8

# 3. (可选) YOLO 训练（imgsz=640 与渲染一致）
yolo train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=128

# 4. 序列准备
# 路径 A：纯 label（-w 8）
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences -w 8

# 路径 B：YOLO 跟踪（需步骤 3）
python scripts/run_track_and_prepare.py -m runs/detect/train/weights/best.pt -d dataset -o sequences

# 5. 行为模型训练（--boost-incorrect 提升错误检测率）
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior \
  --boost-incorrect --aug-multiscale --aug-frame-drop 0.1 --aug-noise 0.02 --epochs 100

# 6. 评估（自动搜索最优 F1 阈值）
python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json

# 7. 演示视频（-d dataset 与训练一致；路径 B 加 -m 权重 --draw-boxes）
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c checkpoints/behavior/best.pt -d dataset -o demo.mp4
```

---

## 1. 数据生成（支持多进程 `-w`）

```bash
python data_generator.py

# 多进程加速
python data_generator.py -b 100 -s 100 -w 8
```

## 2. 渲染与导出（640×640，支持多进程 `-w`）

```bash
python scripts/render_and_export.py --batches batches/ --output dataset --val-ratio 0.2

# 多进程加速
python scripts/render_and_export.py --batches batches/ -o dataset -w 8
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

```bash
# 纯 label（多进程）
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences -w 8
```

```bash
# YOLO 跟踪（GPU 时默认单进程）
python scripts/run_track_and_prepare.py -m runs/detect/train/weights/best.pt -d dataset -o sequences
```

## 6. 行为模型训练（纯网格）

```bash
python scripts/train_behavior.py --data grid --batches batches/ -o checkpoints/behavior --epochs 100
```

## 7. 行为模型训练（序列数据，默认双向 LSTM + 注意力）

```bash
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences
```

```bash
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior
```

**推荐：启用类别平衡与增强**

```bash
# 错误检测率低时使用 --boost-incorrect
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior \
  --boost-incorrect --aug-multiscale --aug-frame-drop 0.1 --aug-noise 0.02

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

```bash
# 从 track_sequences 评估（默认自动搜索最优 F1 阈值）
python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json

# 使用固定阈值
python scripts/eval_behavior.py -c best.pt -d sequences/track_sequences.json --no-threshold-search --incorrect-threshold 0.9

# --reason-override：reason 为错误类时强制 label=incorrect
python scripts/eval_behavior.py -c best.pt -d sequences/track_sequences.json --reason-override

# 从 batches 评估（需 dataset metadata）
python scripts/eval_behavior.py -c best.pt -b batches/ -d dataset
```

# 命令速查（分行）

## 完整训练流程（含 YOLO）

```bash
# 1. 数据生成
python data_generator.py --batches 10 --batch-size 100

# 2. 渲染与导出 YOLO 数据集
python scripts/render_and_export.py --batches batches/ --output dataset --val-ratio 0.2 --skip-n 5

# 3. (可选) YOLO 训练（检测 snake_head/body, food, x2）
yolo train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=600 batch=64

# 4. 序列准备（二选一）
# 路径 A：纯 label，无需 YOLO
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences

# 路径 B：YOLO 跟踪（需先完成步骤 3）
python scripts/run_track_and_prepare.py -m runs/detect/train/weights/best.pt -d dataset -o sequences

# 5. 行为模型训练（推荐启用增强）
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior \
  --class-weights --oversample --aug-multiscale --aug-frame-drop 0.1 --aug-noise 0.02 --epochs 100

# 6. 评估
python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json

# 7. 演示视频（路径 B 可加 -m YOLO权重 --draw-boxes）
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c checkpoints/behavior/best.pt -d dataset -o demo.mp4
```

---

## 1. 数据生成

```bash
python data_generator.py
```

## 2. 渲染与导出

```bash
python scripts/render_and_export.py --batches batches/ --output dataset --val-ratio 0.2 --skip-n 5
```

## 3. 预览标注

```bash
python scripts/preview_labels.py --num 100
```

## 4. YOLO 训练

```bash
yolo train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=128
```

## 5. 序列准备（二选一）

```bash
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences
```

```bash
python scripts/run_track_and_prepare.py -m runs/detect/train/weights/best.pt -d dataset -o sequences
```

## 6. 行为模型训练（纯网格）

```bash
python scripts/train_behavior.py --data grid --batches batches/ -o checkpoints/behavior --epochs 100
```

## 7. 行为模型训练（序列数据）

```bash
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences
```

```bash
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior
```

**推荐：启用类别平衡与增强**

```bash
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior \\
  --class-weights --oversample --aug-multiscale --aug-frame-drop 0.1 --aug-noise 0.02
```

## 8. 行为模型推理

```bash
python scripts/infer_behavior.py --model checkpoints/behavior/best.pt --data sequences/track_sequences.json
```

```bash
python scripts/infer_behavior.py -m checkpoints/behavior/best.pt -d grid --batches batches/
```

## 9. 实战演示视频（YOLO + 行为联合标注）

```bash
# 推荐：指定 dataset，从 metadata 读取实际导出帧，与训练 100% 一致
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c best.pt -d dataset -o demo.mp4

# 无 dataset 时用 --skip-n 模拟（需与 render 参数一致）
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c best.pt -o demo.mp4 --skip-n 5

# YOLO 框 + 行为标注
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -m yolov8n.pt -c best.pt -d dataset -o demo.mp4 --draw-boxes
```

## 10. 链路验证（确保 demo 与训练数据一致）

```bash
# 需先 render_and_export 生成 dataset
python scripts/verify_pipeline.py -b batches/batch_00000.json -e 0 -d dataset
```

## 11. 全量评估（P、R、F1、mAP）

```bash
# 从 track_sequences 评估（与训练格式一致）
python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json

# 从 batches 评估（需 dataset metadata）
python scripts/eval_behavior.py -c best.pt -b batches/ -d dataset
```

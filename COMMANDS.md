# 命令速查（分行）

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

## 8. 行为模型推理

```bash
python scripts/infer_behavior.py --model checkpoints/behavior/best.pt --data sequences/track_sequences.json
```

```bash
python scripts/infer_behavior.py -m checkpoints/behavior/best.pt -d grid --batches batches/
```

## 9. 实战演示视频（YOLO + 行为联合标注）

```bash
# 仅行为标注
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c checkpoints/behavior/best.pt -o demo.mp4

# YOLO 框 + 行为标注
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -m yolov8n.pt -c checkpoints/behavior/best.pt -o demo.mp4 --draw-boxes
```

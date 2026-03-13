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

## 5. YOLO 跟踪与序列准备

```bash
python scripts/run_track_and_prepare.py --model runs/detect/train/weights/best.pt --dataset dataset --output sequences
```

## 6. 行为模型训练（纯网格）

```bash
python scripts/train_behavior.py --data grid --batches batches/ -o checkpoints/behavior --epochs 100
```

## 7. 行为模型训练（YOLO 序列）

```bash
python scripts/run_track_and_prepare.py --model runs/detect/train/weights/best.pt --dataset dataset --output sequences
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

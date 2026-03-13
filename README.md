# behavior_detection

基于贪吃蛇游戏，创建行为识别 + 行为正确性检测的 DEMO，用于生成带标注的视频素材数据。

## 核心算法流程

整体 Pipeline 分为两条可选路径：**纯网格路径**（基于 scene 坐标）和 **视觉路径**（YOLO 检测 + 跟踪）。

```mermaid
flowchart TB
    A["1. 数据生成<br/>data_generator, batch JSON"]
    B["2. 渲染与导出<br/>render_and_export, 图像+YOLO label"]
    C["3. 目标检测+跟踪<br/>YOLO+ByteTrack 可选"]

    A --> B --> C
    A -->|纯网格路径| D
    B -->|视觉路径| D
    C --> D

    D["4. 序列构建 run_track_and_prepare<br/>每蛇每帧12维特征, 输入label或YOLO track"]
    D --> E

    E["5. 行为正确性 train_behavior LSTM<br/>correct/incorrect + reason 共7类"]
    E --> F

    F["6. 实战演示 demo_video<br/>对局渲染, YOLO框+行为标注, 输出MP4"]
```

### 流程说明

| 阶段 | 脚本 | 输入 | 输出 |
|------|------|------|------|
| 1. 数据生成 | `data_generator.py` | 随机种子、规则参数 | `batches/batch_*.json` |
| 2. 渲染导出 | `render_and_export.py` | batch JSON | `dataset/` (images + labels + metadata) |
| 3. 目标检测 | YOLO `train` / `track` | 渲染图像 | 检测框 + track_id |
| 4. 序列构建 | `run_track_and_prepare.py` | dataset 或 batch | `sequences/track_sequences.json` |
| 5. 行为训练 | `train_behavior.py` | sequences 或 grid batches | `checkpoints/behavior/best.pt` |
| 6. 视频演示 | `demo_video.py` | batch + 模型权重 | 带标注的 MP4 视频 |

### 12 维特征（行为模型输入）

每帧每蛇提取：`[head_x, head_y, vel_x, vel_y, food_x, food_y, x2_x, x2_y, has_x2, dist_to_food, dist_to_x2, moving_towards_food]`

---

## 游戏规则

- 无墙壁，蛇撞自己即结束
- 吃 1 个食物 +1 格长度、+1 分
- 每当食物被吃掉时，50% 概率同时生成一个「x2」
- 先吃 x2 再吃食物 → 该食物得 2 分；否则得 1 分
- x2 每波只生效一次，生成新食物时自动失效

## 行为标注

| 标注 | 含义 |
|------|------|
| `correct` / `ate_x2_then_food` | 先吃 x2 再吃食物 ✓ |
| `correct` / `ate_food_no_x2` | 无 x2 时吃食物 ✓ |
| `in_progress` | 进行中（未结束） |
| `incorrect` / `self_collision` | 撞自己 ✗ |
| `incorrect` / `snake_collision` | 蛇间碰撞 ✗ |
| `incorrect` / `x2_wasted` | 先吃食物导致 x2 浪费 ✗ |
| `incorrect` / `timeout` | 超时未吃食物 ✗ |

## 数据格式

batch JSON 结构（支持多蛇）：

```json
{
  "episodes": [
    {
      "scenes": [
        {
          "snakes": [
            { "body": [[x,y],...], "food": [x,y], "x2": [x,y]|null, "score": 0, "color_id": 0 }
          ],
          "step": 0
        }
      ],
      "label": "correct",
      "reason": "ate_food_no_x2",
      "snake_annotations": [
        { "label": "correct", "reason": "ate_food_no_x2" }
      ]
    }
  ]
}
```

每个 `scene` 为一帧状态；`snake_annotations[i]` 为第 i 条蛇的波级标注。

## 使用

### 1. 生成数据

```bash
# 默认：1000 个 batch，每 batch 100 局，输出到 batches/
python data_generator.py

# 自定义参数
python data_generator.py --batches 100 --batch-size 100 --output my_data --mistake-rate 0.2
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `-b, --batches` | 1000 | batch 数量 |
| `-s, --batch-size` | 100 | 每 batch 局数 |
| `-o, --output` | batches | 输出目录 |
| `-m, --mistake-rate` | 0.15 | AI 犯错概率（先吃食物浪费 x2） |
| `-f, --max-foods` | 12 | 每局需吃完的食物波数 |

每个 batch 保存为独立 JSON，如 `batches/batch_00000.json`。每局存储完整场景序列。

### 完整训练流程（含 YOLO）

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

### 2. 回放演示

```bash
pip install pygame
python replay_ui.py
```

回放控制：
- **打开文件 (O)**：点击按钮或按 O 键选择 JSON 数据文件
- **空格**：播放/暂停
- **←/→**：上一帧/下一帧
- **A/D**：上一局/下一局
- **Home/End**：跳到开头/结尾

### 3. 渲染与导出 YOLO 数据集

```bash
python scripts/render_and_export.py --batches batches/ --output dataset --val-ratio 0.2 --skip-n 5
```

输出 `dataset/train`, `dataset/val`（images + labels + metadata.json）。

### 4. 序列准备（二选一）

```bash
# 纯 label：直接从 YOLO label 提取蛇头，无需运行 YOLO
python scripts/run_track_and_prepare.py --from-labels -d dataset -o sequences

# YOLO 跟踪：需先训练 YOLO，再跑 track
python scripts/run_track_and_prepare.py -m yolov8n.pt -d dataset -o sequences
```

### 5. 行为模型训练

```bash
# 纯网格（推荐先验证）
python scripts/train_behavior.py --data grid --batches batches/ -o checkpoints/behavior

# 序列数据
python scripts/train_behavior.py --data sequences/track_sequences.json -o checkpoints/behavior
```

### 6. 实战演示视频

```bash
# 行为标注
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -c checkpoints/behavior/best.pt -o demo.mp4

# YOLO + 行为联合标注
python scripts/demo_video.py -b batches/batch_00000.json -e 0 -m yolov8n.pt -c checkpoints/behavior/best.pt -o demo.mp4 --draw-boxes
```

### 7. 全量评估（P、R、F1、mAP）

```bash
# 从 track_sequences（推荐，与训练格式一致）
python scripts/eval_behavior.py -c checkpoints/behavior/best.pt -d sequences/track_sequences.json

# 从 batches
python scripts/eval_behavior.py -c best.pt -b batches/ -d dataset
```

### 8. 代码调用

```python
from data_generator import generate_dataset, run_episode

# 生成数据：100 个 batch，每 batch 100 局
generate_dataset(num_batches=100, batch_size=100, output_dir="batches")

# 单局运行（含 AI 犯错概率）
ep = run_episode(seed=42, max_foods=12, ai_mistake_rate=0.15)
print(ep["label"], ep["reason"], len(ep["scenes"]))
```

---

## 依赖

| 用途 | 包 |
|------|-----|
| 游戏 / 回放 / 渲染 | `pygame` |
| 目标检测 / 跟踪 | `ultralytics` (YOLO) |
| 行为模型 | `torch` |
| 视频输出 | `opencv-python` |

---

## 项目结构

```
behavior_detection/
├── data_generator.py       # 数据生成
├── game.py, ai.py          # 游戏逻辑与 AI
├── replay_ui.py            # 回放演示
├── scripts/
│   ├── render_and_export.py   # 渲染 → YOLO 数据集
│   ├── preview_labels.py      # 标注预览
│   ├── run_track_and_prepare.py  # 序列构建
│   ├── train_behavior.py       # 行为模型训练
│   ├── infer_behavior.py       # 行为推理
│   └── demo_video.py           # 实战演示视频
├── models/
│   └── behavior_correctness.py # LSTM 行为/正确性模型
├── batches/                 # 生成的对局数据
├── dataset/                 # 渲染输出的 YOLO 数据集
├── sequences/               # 序列特征
└── checkpoints/behavior/    # 模型权重
```

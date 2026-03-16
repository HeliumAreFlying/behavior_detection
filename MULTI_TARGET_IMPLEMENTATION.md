# 多目标行为正确性检测 - 开源框架实现方案

## 一、整体 Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ 渲染帧/截图  │ -> │ 目标检测 (YOLO)   │ -> │ 多目标跟踪       │ -> │ 行为+正确性分类   │
│ 或 grid 状态 │    │ snake/food/x2    │    │ 区分蛇1/蛇2/蛇3   │    │ 每蛇独立标注      │
└─────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

**两条可选路径：**

| 路径 | 输入 | 优势 | 适用场景 |
|------|------|------|----------|
| **A. 纯视觉** | 渲染出的 RGB 图像 | 真实、可迁移到屏幕录像 | 模拟器/录屏输入 |
| **B. 网格状态** | 已有 scene（grid 坐标） | 无需检测、训练简单 | 本数据生成器、可作 baseline |

**推荐**：先做 B 验证算法，再做 A 实现端到端视觉检测。

---

## 二、开源框架选型（全栈）

| 环节 | 推荐框架 | 安装 | 用途 |
|------|----------|------|------|
| **目标检测** | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | `pip install ultralytics` | 检测 snake_head/body, food, x2 |
| **多目标跟踪** | [ByteTrack](https://github.com/ifzhang/ByteTrack) 或 [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | 配合 YOLO 使用 | 跨帧关联同一蛇 |
| **时序/分类** | [PyTorch](https://pytorch.org/) + LSTM/Transformer | `pip install torch` | 行为识别 + 正确性分类 |

**说明**：
- Ultralytics YOLO 自带 `model.track()` 接口，可输出带 track_id 的检测框，底层已集成 ByteTrack，**优先使用此方案**，无需单独安装 ByteTrack。
- 若需要更精细的 ReID 特征，可考虑 BoT-SORT；当前场景蛇颜色固定，YOLO 内置跟踪通常足够。

---

## 三、多蛇场景的特殊设计

### 3.1 检测类别设计

**不区分颜色**，仅 4 类，由**目标跟踪 (ByteTrack)** 区分不同蛇/人：

| class_id | 类别名 | 说明 |
|----------|--------|------|
| 0 | snake_head | 蛇头 |
| 1 | snake_body | 蛇身 |
| 2 | food | 食物 |
| 3 | x2 | 加倍道具 |

跟踪为每个检测框分配 `track_id`，同一蛇在不同帧保持相同 ID，从而实现多人区分。

### 3.2 跟踪策略

- YOLO 检测 4 类，输出 bbox。
- 调用 `model.track(persist=True)`，ByteTrack 为每个目标分配 `track_id`。
- 蛇头 track_id 对应身份，蛇身/食物/x2 可通过位置关联到对应蛇。

### 3.3 行为正确性标签

- 与每张图像一一对应，存放在 `train/behavior/`, `val/behavior/` 下
- 文件名与图像一致：`000000.json` 对应 `000000.png`
- 格式：`{"snake_annotations": [{"label": "correct", "reason": "in_progress"}, ...]}`
- `snake_annotations[i]` 为第 i 条蛇的标注（按 scene 中蛇的顺序）
- 跟踪后需将 `track_id` 与 `snake_annotations` 下标对应（如首帧位置匹配）

---

## 四、实现步骤（分阶段）

### 阶段 1：数据与渲染（基于现有 data_generator）

1. **渲染脚本**：从 `scene` 生成图像 + bbox
   - 输入：`scene`（含 snakes, step, snake_annotations）
   - 输出：图像文件 + YOLO 格式 txt（`class_id x_center y_center w h` 归一化）

2. **扩展 data_generator**
   - 输出每帧 `bboxes`（像素坐标，与 replay_ui 渲染参数一致）
   - 或单独写 `render_and_export.py`：读取 batch JSON → 调用 game/replay 渲染 → 导出图像和标注

3. **数据集划分**
   - `render_and_export` 输出 `dataset/train`、`dataset/val`，按**局（episode）**划分（约 val_ratio 的局进 val），保证验证集样本量合理
   - `run_track_and_prepare` 生成的 `track_sequences.json` 中每条序列带 `split`，与 dataset 一致；训练与评估按此划分，评估默认仅 val 集
   - 每帧行为 JSON 含 `head_forward_type`：蛇头前方一格 0=空/1=己身/2=他蛇身体/3=他蛇头，作为模型输入
   - 每集可有多蛇，每蛇有独立 `snake_annotations`

### 阶段 2：目标检测

```bash
pip install ultralytics
```

1. 准备 `data.yaml`：
```yaml
path: ./dataset
train: images/train
val: images/val
nc: 4
names: ['snake_head', 'snake_body', 'food', 'x2']
```

2. 训练：
```bash
yolo train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

3. 推理 + 跟踪（视频/图像序列）：
```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
results = model.track(source="frames/", persist=True)  # 输出带 track_id
```

### 阶段 3：行为 + 正确性（时序模型）

**输入**：每蛇每帧特征向量，如  
`[head_x, head_y, food_x, food_y, x2_x, x2_y, has_x2, score_delta, ...]`

**输出**：
- 行为：moving / ate_x2 / ate_food / ate_food_x2 / collision / timeout
- 正确性：correct / incorrect
- 原因：ate_x2_then_food / x2_wasted / ate_food_no_x2 / ...

**模型**：LSTM 或 1D CNN + 分类头

```python
import torch
import torch.nn as nn

class BehaviorCorrectnessModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_behavior = nn.Linear(hidden_dim, num_classes)  # 行为类别
        self.fc_correct = nn.Linear(hidden_dim, 2)            # correct/incorrect
        self.fc_reason = nn.Linear(hidden_dim, 6)             # reason 类别

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时刻
        behavior = self.fc_behavior(out)
        correct = self.fc_correct(out)
        reason = self.fc_reason(out)
        return behavior, correct, reason
```

**数据**：现有 batch JSON 已包含 `snake_annotations`，可从 scene 序列构造特征，无需额外标注。

### 阶段 4：端到端串联

1. 图像序列 → YOLO 检测 + 跟踪 → 每帧每蛇的 (head, food, x2) 坐标
2. 按 track_id / class_id 分组，构造每蛇的时序特征
3. 送入 LSTM 模型 → 输出每蛇的 label + reason
4. 与 `snake_annotations` 计算 loss，训练

---

## 五、快速验证：纯网格路径（无需 YOLO）

若暂时不做视觉检测，可直接用 **scene 里的 grid 坐标** 作为输入：

1. 从 `scene["snakes"][i]` 读取 body[0], food, x2, score
2. 组成 `(T, D)` 序列，D=8 左右
3. 训练 LSTM 预测 `snake_annotations[i]` 的 label + reason

这样可验证：
- 特征设计是否合理
- 模型容量是否足够
- 正确性分类是否可学习

再在此基础上接入 YOLO 检测，替换 grid 为检测框中心点即可。

---

## 六、依赖与目录建议

```txt
# requirements.txt 扩展
ultralytics>=8.0
torch>=2.0
```

```
behavior_detection/
├── data_generator.py      # 已有
├── game.py, ai.py         # 已有
├── replay_ui.py           # 已有
├── scripts/
│   ├── render_and_export.py   # 渲染 batch → YOLO 数据集 + 行为标签
│   └── preview_labels.py      # 绘制标注预览 100 张
├── dataset/               # 渲染输出
│   ├── train/images, train/labels, train/behavior
│   ├── val/images, val/labels, val/behavior
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── scripts/
│   ├── render_and_export.py   # 渲染 + 导出 YOLO 标注
│   ├── train_detector.py      # YOLO 训练脚本
│   └── train_behavior.py      # LSTM 行为/正确性训练
└── models/
    └── behavior_correctness.py
```

---

## 七、小结

| 任务 | 开源方案 | 备注 |
|------|----------|------|
| 目标检测 | Ultralytics YOLO | 按颜色拆类，便于多蛇 |
| 多目标跟踪 | YOLO 内置 `track()` | 或 ByteTrack/BoT-SORT |
| 行为识别 | PyTorch + LSTM | 输入为每蛇时序特征 |
| 正确性检测 | 同上，多任务输出 | label + reason |

**优先建议**：先实现 `render_and_export.py` 和基于 grid 的 LSTM 训练，验证流程后再接入 YOLO 检测。

---

## 八、快速使用

```bash
# 1. 生成 batch 数据（若尚未生成）
python data_generator.py --batches 10 --batch-size 50

# 2. 渲染为 YOLO 数据集（--skip-n 5：非关键帧每 5 帧采样，关键帧全保留）
python scripts/render_and_export.py --output dataset --skip-n 5

# 3. 生成 100 张标注预览
python scripts/preview_labels.py --num 100
# 输出到 preview/ 目录，可打开查看
```

### 行为标签结构

每张图像对应一个 `behavior/xxx.json`：

```json
{
  "snake_annotations": [
    {"label": "correct", "reason": "in_progress"},
    {"label": "incorrect", "reason": "x2_wasted"}
  ]
}
```

- `snake_annotations[i]`：第 i 条蛇（scene 中 snakes[i]）针对**当前食物**的标注
- `label`：correct / incorrect
- `reason`：ate_x2_then_food / ate_food_no_x2 / in_progress / x2_wasted / timeout / self_collision / snake_collision

训练时：YOLO+跟踪得到每蛇轨迹 → 按 episode 串成序列 → 读取对应 behavior/*.json 作为监督信号

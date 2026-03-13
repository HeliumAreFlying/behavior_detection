# 行为识别 + 正确性检测 模型训练方案

> **多目标（多蛇）完整实现方案** 见 → [MULTI_TARGET_IMPLEMENTATION.md](./MULTI_TARGET_IMPLEMENTATION.md)

## 零、开源方案选型

### 目标检测

| 方案 | 安装 | 特点 | 推荐度 |
|------|------|------|--------|
| **Ultralytics YOLO** | `pip install ultralytics` | 安装简单、文档全、YOLOv8/v11 成熟，支持自定义数据集 | ⭐⭐⭐⭐⭐ |
| MMDetection | 依赖 MMCV，配置较复杂 | OpenMMLab 生态，模型多 | ⭐⭐⭐ |
| Detectron2 | 需编译 | Facebook 出品，研究常用 | ⭐⭐⭐ |

**推荐：Ultralytics YOLO（YOLOv8 或 YOLOv11）**
- 安装：`pip install ultralytics`
- 支持 YOLO 格式标注（每图一个 txt：`class_id x_center y_center w h` 归一化）
- 训练：`yolo train data=your_data.yaml model=yolov8n.pt`

### 行为识别（时序 / 视频）

| 方案 | 适用场景 | 安装 | 特点 | 推荐度 |
|------|----------|------|------|--------|
| **PyTorch + LSTM/Transformer** | 序列特征分类（本场景首选） | `pip install torch` | 轻量、可完全自定义、适合 grid 状态序列 | ⭐⭐⭐⭐⭐ |
| **MMAction2** | 真实视频动作识别 | 依赖 MMCV，配置多 | 3D CNN、双流等，适合 RGB 视频 | ⭐⭐⭐ |
| **PaddleVideo** | 视频理解 | PaddlePaddle | 国产生态 | ⭐⭐⭐ |

**本场景推荐：PyTorch 自建时序模型**

你的数据是「网格状态序列」，不是 RGB 视频，不必用 MMAction2 这类重框架。建议：

1. **输入**：每帧提取特征（如 snake_head_pos, food_pos, x2_pos, score 等）→ 组成 T×D 序列
2. **模型**：LSTM 或 1D CNN + Transformer 做序列分类
3. **类别**：moving / ate_x2 / ate_food / ate_food_x2 / collision

若坚持用「图像序列」做行为识别，可考虑 MMAction2，但计算量和数据量需求会大很多。

### 正确性检测

- 在行为序列特征上接分类头即可
- 可与行为识别共享 backbone，多任务学习
- 或：先跑行为识别得到 event 序列，再用规则 / 小 MLP 判断 correct/incorrect

### 选型小结

| 任务 | 推荐方案 | 安装 |
|------|----------|------|
| 目标检测 | Ultralytics YOLO | `pip install ultralytics` |
| 行为识别 | PyTorch + LSTM/Transformer | `pip install torch` |
| 正确性检测 | 复用行为特征 + 分类头 | 同上 |

---

## 一、目标分解

| 任务 | 输入 | 输出 | 标注层级 |
|------|------|------|----------|
| **行为识别** | 时序 / 单帧 | 当前在做什么（moving/eating_x2/eating_food/...） | 时间段 + 单帧 |
| **正确性检测** | 时序 / 行为序列 | 本段行为是否正确 | 时间段（波级） |
| **目标检测** | 单帧图像 | 目标框 + 类别（snake_head, snake_body, food, x2） | 单帧 |

---

## 二、当前规则是否足够？

### 2.1 已有数据

- **单帧**：`snake`, `food`, `x2` 的网格坐标 → 可转为像素坐标 + 目标框
- **事件**：`ate_x2`, `ate_food`, `ate_food_x2`, `continue`, `collision`（可从相邻帧推断）
- **波级标注**：每波结束有 correct/incorrect + reason

### 2.2 结论：**当前规则已能支撑两个任务**

- **行为识别**：能从 scene 序列推断出「moving / eating_x2 / eating_food / collision」
- **正确性检测**：波级 label + reason 已足够
- **目标检测**：有 snake/food/x2 的精确坐标，可生成 bbox

**不必大改游戏规则**，只需在数据格式和标注粒度上做扩展。

---

## 三、建议的数据扩展

### 3.1 单帧层面：目标检测标注

在 `scene` 中增加 `bboxes`（像素坐标），便于直接用于目标检测：

```json
{
  "snake": [[7,7],[6,7],[5,7]],
  "food": [3,5],
  "x2": [10,2],
  "bboxes": [
    {"class": "snake_head", "xyxy": [280,280,320,320]},
    {"class": "snake_body", "xyxy": [240,280,280,320]},
    {"class": "food", "xyxy": [120,200,160,240]},
    {"class": "x2", "xyxy": [400,80,440,120]}
  ]
}
```

`xyxy` 由 `(grid_x * cell_size, grid_y * cell_size)` 等计算得到。

### 3.2 时序层面：帧级行为标签

在每帧或每步记录「本步发生的事件」，用于行为识别训练：

```json
{
  "step": 42,
  "event": "moving",
  "event_detail": "toward_food"
}
```

事件类型建议：

| event | 含义 | 正确性 |
|-------|------|--------|
| `moving` | 普通移动 | 依赖上下文 |
| `ate_x2` | 吃到 x2 | 正确（本步） |
| `ate_food` | 吃到食物（无 x2 加成） | 正确（无 x2 时）/ 错误（有 x2 时=x2_wasted） |
| `ate_food_x2` | 吃到食物（有 x2 加成） | 正确 |
| `collision` | 撞自己 | 错误 |
| `timeout` | 本波超时 | 错误 |

### 3.3 波级标注（已有）

- `wave_id`：当前属于第几波
- `wave_label`：本波 correct/incorrect
- `wave_reason`：ate_x2_then_food / x2_wasted / timeout / ...

---

## 四、推荐训练流程

```
1. 目标检测
   - 输入：单帧（或由 scene 渲染的图像）
   - 标注：snake_head, snake_body, food, x2 的 bbox
   - 模型：YOLO / Faster R-CNN 等

2. 行为识别（时序）
   - 输入：连续 N 帧（或 N 帧的目标特征）
   - 标注：每帧 event（moving/ate_x2/ate_food/...）
   - 模型：3D CNN / LSTM / Transformer

3. 正确性检测
   - 输入：一段行为序列（如一波内的帧）
   - 标注：波级 correct/incorrect + reason
   - 模型：在行为特征上接分类头，或规则 + 轻量模型
```

### 4.1 错误检测率提升（已实现）

当 incorrect 类召回率较低时，可使用：

- **训练**：`--boost-incorrect`（启用 Focal Loss、类别权重、过采样、incorrect 2× 权重）
- **推理/评估**：`--incorrect-threshold 0.3`（降低判定阈值）、`--reason-override`（reason 为错误类时强制 incorrect）

---

## 五、是否要丰富游戏规则？

### 5.1 当前规则已覆盖的内容

- 正确：先吃 x2 再吃食物、无 x2 时吃食物
- 错误：x2 浪费、超时、撞自己
- 目标：蛇头、蛇身、食物、x2

### 5.2 可选扩展（非必须）

| 扩展 | 作用 | 优先级 |
|------|------|--------|
| 蛇头朝向 | 辅助判断「是否朝目标移动」 | 低 |
| 墙壁/障碍 | 增加空间复杂度 | 低 |
| 多种道具 | 行为类别更丰富 | 低 |

建议：**先不扩展规则**，用现有数据完成：
- 目标检测（4 类）
- 行为识别（5–6 类 event）
- 正确性检测（波级 2 类 + reason）

验证 pipeline 后再考虑加规则。

---

## 六、落地建议

1. **扩展 data_generator**：输出 `bboxes`、`event`、`wave_id` 等字段。
2. **渲染脚本**：从 scene 生成图像，配合 bbox 生成 COCO/YOLO 等格式。
3. **按任务拆分数据集**：目标检测 / 行为识别 / 正确性检测 各自的数据格式与划分。

需要的话，我可以直接给出 `data_generator` 和渲染脚本的修改示例代码。

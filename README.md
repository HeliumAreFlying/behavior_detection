# behavior_detection

基于贪吃蛇游戏，创建行为识别+行为正确性检测的 DEMO，用于生成带标注的视频素材数据。

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
| `incorrect` / `self_collision` | 撞自己 ✗ |
| `incorrect` / `x2_wasted` | 先吃食物导致 x2 浪费 ✗ |
| `incorrect` / `timeout` | 超时未吃食物 ✗ |

## 数据格式

输出为 JSON 文件，结构如下：

```json
{
  "episodes": [
    {
      "scenes": [
        { "snake": [[x,y],...], "food": [x,y], "x2": [x,y]|null, "score": 0, "x2_active": false, "step": 0 }
      ],
      "label": "correct",
      "reason": "ate_food_no_x2",
      "foods_eaten": 5
    }
  ]
}
```

每个 `scene` 为一帧状态，用于后续行为识别；无需实际渲染视频。

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

每个 batch 保存为独立 JSON，如 `batches/batch_00000.json`。每局存储完整场景序列。

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

### 3. 代码调用

```python
from data_generator import generate_dataset, run_episode

# 生成数据：100 个 batch，每 batch 100 局
generate_dataset(num_batches=100, batch_size=100, output_dir="batches")

# 单局运行（含 AI 犯错概率）
ep = run_episode(seed=42, max_foods=5, ai_mistake_rate=0.15)
print(ep["label"], ep["reason"], len(ep["scenes"]))
```

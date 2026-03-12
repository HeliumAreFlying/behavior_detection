"""
数据生成器：运行贪吃蛇AI，记录场景序列，输出带标注的JSON
"""

import json
from pathlib import Path
from game import SnakeGame, Direction
from ai import choose_direction


# 正确/错误行为原因
LABEL_CORRECT = "correct"
LABEL_INCORRECT = "incorrect"

REASON_ATE_X2_THEN_FOOD = "ate_x2_then_food"    # 先吃x2再吃食物 ✓
REASON_ATE_FOOD_NO_X2 = "ate_food_no_x2"        # 无x2时吃食物 ✓
REASON_SELF_COLLISION = "self_collision"        # 撞自己 ✗
REASON_X2_WASTED = "x2_wasted"                  # 先吃食物导致x2浪费 ✗
REASON_TIMEOUT = "timeout"                      # 超时未吃 ✗


def run_episode(
    width: int = 15,
    height: int = 15,
    max_steps_per_food: int = 80,
    max_foods: int = 5,
    seed: int | None = None,
    ai_randomness: float = 0.35,
    ai_mistake_rate: float = 0.15,
) -> dict:
    """
    运行一局游戏，返回场景序列和标注
    每吃到一定数量食物或结束时停止；超时步数内未吃食物则判为错误
    """
    game = SnakeGame(width=width, height=height, seed=seed)

    scenes: list[dict] = []
    foods_eaten = 0
    steps_since_food = 0
    x2_was_wasted = False
    used_x2_successfully = False  # 是否曾有「先吃x2再吃食物」

    scenes.append(game.get_state().to_dict())

    while foods_eaten < max_foods:
        if steps_since_food >= max_steps_per_food:
            scenes.append(game.get_state().to_dict())
            return {
                "scenes": scenes,
                "label": LABEL_INCORRECT,
                "reason": REASON_TIMEOUT,
                "foods_eaten": foods_eaten,
            }

        had_x2_before_move = game.x2 is not None
        direction = choose_direction(
            game, randomness=ai_randomness, mistake_rate=ai_mistake_rate
        )
        alive, event = game.move(direction)
        scenes.append(game.get_state().to_dict())
        steps_since_food += 1

        if not alive:
            return {
                "scenes": scenes,
                "label": LABEL_INCORRECT,
                "reason": REASON_SELF_COLLISION,
                "foods_eaten": foods_eaten,
            }

        if event == "ate_x2":
            steps_since_food = 0
        elif event == "ate_food":
            if had_x2_before_move:
                x2_was_wasted = True  # 先吃食物，x2被浪费
            foods_eaten += 1
            steps_since_food = 0
        elif event == "ate_food_x2":
            used_x2_successfully = True
            foods_eaten += 1
            steps_since_food = 0

    if x2_was_wasted:
        return {
            "scenes": scenes,
            "label": LABEL_INCORRECT,
            "reason": REASON_X2_WASTED,
            "foods_eaten": foods_eaten,
        }

    return {
        "scenes": scenes,
        "label": LABEL_CORRECT,
        "reason": REASON_ATE_X2_THEN_FOOD if used_x2_successfully else REASON_ATE_FOOD_NO_X2,
        "foods_eaten": foods_eaten,
    }


def generate_dataset(
    num_batches: int = 1000,
    batch_size: int = 100,
    output_dir: str | Path = "batches",
    **kwargs,
) -> None:
    """
    生成数据，每 batch_size 局存一个 JSON 文件（一个 batch）
    num_batches: 生成多少个 batch
    batch_size: 每个 batch 的局数
    output_dir: 输出目录
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = num_batches * batch_size

    for b in range(num_batches):
        episodes = []
        base_seed = b * batch_size
        for i in range(batch_size):
            ep = run_episode(seed=base_seed + i, **kwargs)
            episodes.append(ep)

        batch_path = out / f"batch_{b:05d}.json"
        data = {"episodes": episodes, "batch_id": b, "batch_size": len(episodes)}
        batch_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"Batch {b + 1}/{num_batches} -> {batch_path}")

    print(f"完成：共 {total} 局，{num_batches} 个 batch -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="生成贪吃蛇行为识别数据集")
    p.add_argument("--batches", "-b", type=int, default=1000, help="batch 数量")
    p.add_argument("--batch-size", "-s", type=int, default=100, help="每 batch 局数")
    p.add_argument("--output", "-o", default="batches", help="输出目录")
    p.add_argument("--mistake-rate", "-m", type=float, default=0.15,
                   help="AI 犯错概率（先吃食物浪费 x2）")
    args = p.parse_args()
    generate_dataset(
        num_batches=args.batches,
        batch_size=args.batch_size,
        output_dir=args.output,
        ai_mistake_rate=args.mistake_rate,
    )

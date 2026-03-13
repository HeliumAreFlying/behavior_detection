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
REASON_ATE_FOOD_NO_X2 = "ate_food_no_x2"        # 吃食物 ✓
REASON_SELF_COLLISION = "self_collision"        # 撞自己 ✗
REASON_SNAKE_COLLISION = "snake_collision"      # 蛇间碰撞 ✗
REASON_X2_WASTED = "x2_wasted"                  # 先吃食物导致x2浪费 ✗
REASON_TIMEOUT = "timeout"                      # 超时未吃 ✗
REASON_IN_PROGRESS = "in_progress"              # 进行中


def _annot_for_eat(event: str, had_x2: bool) -> tuple[str, str]:
    """根据吃食物事件返回 (label, reason)"""
    if event == "ate_food_x2":
        return LABEL_CORRECT, REASON_ATE_X2_THEN_FOOD
    if event == "ate_food" and had_x2:
        return LABEL_INCORRECT, REASON_X2_WASTED
    return LABEL_CORRECT, REASON_ATE_FOOD_NO_X2


def run_episode(
    width: int = 15,
    height: int = 15,
    max_steps_per_food: int = 80,
    max_foods: int = 12,
    seed: int | None = None,
    ai_randomness: float = 0.18,
    ai_mistake_rate: float = 0.15,
    multi_snake_prob: float = 0.5,
) -> dict:
    """
    运行一局游戏，支持多蛇
    multi_snake_prob: 50% 概率 2~3 条蛇，否则 1 条
    """
    rng = __import__("random").Random(seed)
    num_snakes = 1 if rng.random() >= multi_snake_prob else rng.randint(2, 3)
    game = SnakeGame(width=width, height=height, num_snakes=num_snakes, seed=seed)

    scenes: list[dict] = []
    waves_done = [0] * num_snakes
    steps_since_food = [0] * num_snakes
    snake_first_error: list[str | None] = [None] * num_snakes
    snake_last_correct: list[str] = [REASON_ATE_FOOD_NO_X2] * num_snakes

    def attach_scene_annotations(sc: dict, eat_event: str | None, eat_snake: int | None,
                                 had_x2: list[bool] | None, timeout_snakes: set[int] | None) -> None:
        """在生成时预计算并附加当前帧的每条蛇标注（针对当前食物）"""
        anns = []
        for i in range(num_snakes):
            if timeout_snakes and i in timeout_snakes:
                anns.append({"label": LABEL_INCORRECT, "reason": REASON_TIMEOUT})
            elif eat_snake is not None and i == eat_snake and eat_event:
                lbl, rsn = _annot_for_eat(eat_event, had_x2[i] if had_x2 else False)
                anns.append({"label": lbl, "reason": rsn})
            else:
                anns.append({"label": LABEL_CORRECT, "reason": REASON_IN_PROGRESS})
        sc["snake_annotations"] = anns

    state = game.get_state().to_dict()
    attach_scene_annotations(state, None, None, None, None)
    scenes.append(state)

    def build_snake_annotations() -> list[dict]:
        return [
            {
                "label": LABEL_INCORRECT if r else LABEL_CORRECT,
                "reason": r or snake_last_correct[i],
            }
            for i, r in enumerate(snake_first_error)
        ]

    while min(waves_done) < max_foods:
        any_timeout = False
        timeout_snakes: set[int] = set()
        for i in range(num_snakes):
            if steps_since_food[i] >= max_steps_per_food:
                if snake_first_error[i] is None:
                    snake_first_error[i] = REASON_TIMEOUT
                waves_done[i] += 1
                game.respawn_food_for(i)
                steps_since_food[i] = 0
                any_timeout = True
                timeout_snakes.add(i)
        if any_timeout:
            state = game.get_state().to_dict()
            attach_scene_annotations(state, None, None, None, timeout_snakes)
            scenes.append(state)
            continue

        had_x2 = [game.snakes[i]["x2"] is not None for i in range(num_snakes)]
        directions = [
            choose_direction(game, i, randomness=ai_randomness, mistake_rate=ai_mistake_rate)
            for i in range(num_snakes)
        ]
        alive, event, evt_data = game.move_all(directions)
        if alive:
            state = game.get_state().to_dict()
            eat_evt = event if event in ("ate_food", "ate_food_x2") else None
            eat_snk = evt_data if isinstance(evt_data, int) else None
            attach_scene_annotations(state, eat_evt, eat_snk, had_x2, None)
            scenes.append(state)

        for i in range(num_snakes):
            steps_since_food[i] += 1

        if not alive:
            colliding: set[int] = set()
            if event == "snake_collision" and isinstance(evt_data, (list, tuple)):
                colliding = set(evt_data)
            elif event == "self_collision" and isinstance(evt_data, int):
                colliding = {evt_data}
            for i in colliding:
                snake_first_error[i] = REASON_SNAKE_COLLISION if event == "snake_collision" else REASON_SELF_COLLISION
            # 碰撞不追加新帧，但最后一帧需更新标注（显示碰撞结果）
            reason = REASON_SNAKE_COLLISION if event == "snake_collision" else REASON_SELF_COLLISION
            for i in colliding:
                scenes[-1]["snake_annotations"][i] = {"label": LABEL_INCORRECT, "reason": reason}
            return {
                "scenes": scenes,
                "label": LABEL_INCORRECT,
                "reason": REASON_SNAKE_COLLISION if event == "snake_collision" else REASON_SELF_COLLISION,
                "foods_eaten": sum(waves_done),
                "num_snakes": num_snakes,
                "snake_annotations": build_snake_annotations(),
            }

        evt_snake = evt_data if isinstance(evt_data, int) else None
        if event == "ate_x2" and evt_snake is not None:
            steps_since_food[evt_snake] = 0
        elif event in ("ate_food", "ate_food_x2") and evt_snake is not None:
            i = evt_snake
            waves_done[i] += 1
            steps_since_food[i] = 0
            if had_x2[i]:
                if snake_first_error[i] is None:
                    snake_first_error[i] = REASON_X2_WASTED
            else:
                snake_last_correct[i] = REASON_ATE_FOOD_NO_X2
            if event == "ate_food_x2":
                snake_last_correct[i] = REASON_ATE_X2_THEN_FOOD

    first_err = next((r for r in snake_first_error if r is not None), None)
    return {
        "scenes": scenes,
        "label": LABEL_INCORRECT if first_err else LABEL_CORRECT,
        "reason": first_err or snake_last_correct[-1],
        "foods_eaten": sum(waves_done),
        "num_snakes": num_snakes,
        "snake_annotations": build_snake_annotations(),
    }


def _pool_join_timeout(pool, timeout: float = 3.0) -> None:
    """等待 Pool 子进程退出，每个进程最多等待 timeout 秒，避免 join() 无限阻塞"""
    import time
    if not hasattr(pool, "_pool"):
        return
    t0 = time.perf_counter()
    for p in pool._pool:
        remain = max(0, timeout - (time.perf_counter() - t0))
        if remain > 0 and p.is_alive():
            p.join(timeout=remain)
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)


def _generate_one_batch(args: tuple) -> int:
    """生成单个 batch（子进程调用）"""
    batch_id, batch_size, output_dir, kwargs = args
    episodes = []
    base_seed = batch_id * batch_size
    for i in range(batch_size):
        ep = run_episode(seed=base_seed + i, **kwargs)
        episodes.append(ep)
    out = Path(output_dir)
    batch_path = out / f"batch_{batch_id:05d}.json"
    data = {"episodes": episodes, "batch_id": batch_id, "batch_size": len(episodes)}
    batch_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return batch_id


def generate_dataset(
    num_batches: int = 1000,
    batch_size: int = 100,
    output_dir: str | Path = "batches",
    workers: int | None = None,
    **kwargs,
) -> None:
    """
    生成数据，每 batch_size 局存一个 JSON 文件（一个 batch）
    num_batches: 生成多少个 batch
    batch_size: 每个 batch 的局数
    output_dir: 输出目录
    workers: 并行进程数，1 或 None 表示单进程
    """
    import multiprocessing as mp

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = num_batches * batch_size

    workers = workers or 1
    workers = min(workers, num_batches)

    if workers <= 1:
        for b in range(num_batches):
            _generate_one_batch((b, batch_size, str(out), kwargs))
            print(f"Batch {b + 1}/{num_batches} -> {out / f'batch_{b:05d}.json'}")
    else:
        work_items = [(b, batch_size, str(out), kwargs) for b in range(num_batches)]
        print(f"使用 {workers} 个进程并行生成...")
        pool = mp.Pool(processes=workers)
        interrupted = False
        try:
            for idx, batch_id in enumerate(
                pool.imap_unordered(_generate_one_batch, work_items, chunksize=1)
            ):
                print(f"Batch {batch_id + 1}/{num_batches} -> {out / f'batch_{batch_id:05d}.json'}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\n[中断] 用户取消，正在终止工作进程...", flush=True)
            pool.terminate()
            _pool_join_timeout(pool, timeout=3.0)
            raise SystemExit(130)
        finally:
            if interrupted:
                pass  # 已在 except 中 terminate + _pool_join_timeout，主进程退出时 OS 会回收
            else:
                pool.close()
                pool.join()

    print(f"完成：共 {total} 局，{num_batches} 个 batch -> {out}")


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    p = argparse.ArgumentParser(description="生成贪吃蛇行为识别数据集")
    p.add_argument("--batches", "-b", type=int, default=10, help="batch 数量")
    p.add_argument("--batch-size", "-s", type=int, default=10, help="每 batch 局数")
    p.add_argument("--output", "-o", default="batches", help="输出目录")
    p.add_argument("--workers", "-w", type=int, default=None,
                   help="并行进程数，默认 CPU 核心数")
    p.add_argument("--mistake-rate", "-m", type=float, default=0.15,
                   help="AI 犯错概率（先吃食物浪费 x2）")
    p.add_argument("--max-foods", "-f", type=int, default=12,
                   help="每局需吃完的食物波数")
    args = p.parse_args()
    workers = args.workers or (mp.cpu_count() or 4)
    generate_dataset(
        num_batches=args.batches,
        batch_size=args.batch_size,
        output_dir=args.output,
        workers=workers,
        ai_mistake_rate=args.mistake_rate,
        max_foods=args.max_foods,
    )

"""
贪吃蛇AI：带随机性，保证不自杀
"""

import random
from game import SnakeGame, Direction


def get_safe_directions(game: SnakeGame) -> list[Direction]:
    """返回所有不会撞到自己的方向"""
    head = game.snake[0]
    body_set = set(tuple(p) for p in game.snake[1:])
    safe = []
    for d in Direction:
        dx, dy = d.value
        nx = (head[0] + dx) % game.width
        ny = (head[1] + dy) % game.height
        if (nx, ny) not in body_set:
            safe.append(d)
    return safe


def manhattan(a: list[int], b: list[int], w: int, h: int) -> int:
    """环形网格上的曼哈顿距离（近似）"""
    dx = min(abs(a[0] - b[0]), w - abs(a[0] - b[0]))
    dy = min(abs(a[1] - b[1]), h - abs(a[1] - b[1]))
    return dx + dy


def choose_direction(
    game: SnakeGame,
    randomness: float = 0.35,
    mistake_rate: float = 0.0,
) -> Direction:
    """
    选择移动方向
    randomness: 0~1，随机选安全方向的比例
    mistake_rate: 0~1，有 x2 时故意先吃食物的概率（犯错）
    """
    safe = get_safe_directions(game)
    if not safe:
        return Direction.RIGHT  # 无安全方向时随便返回

    # 随机探索
    if random.random() < randomness:
        return random.choice(safe)

    # 贪心选目标，有 mistake_rate 时会故意忽略 x2 先吃食物
    head = game.snake[0]
    targets = []
    if game.x2 and not game.x2_active:
        if random.random() >= mistake_rate:
            targets.append(game.x2)
    targets.append(game.food)

    best_d = safe[0]
    best_dist = float("inf")
    for d in safe:
        dx, dy = d.value
        nx = (head[0] + dx) % game.width
        ny = (head[1] + dy) % game.height
        npos = [nx, ny]
        for t in targets:
            dist = manhattan(npos, t, game.width, game.height)
            if dist < best_dist:
                best_dist = dist
                best_d = d
                break

    return best_d

"""
贪吃蛇AI：带随机性，避免撞自己和撞其他蛇
"""

import random
from game import SnakeGame, Direction


def get_safe_directions(game: SnakeGame, snake_idx: int) -> list[Direction]:
    """返回不会撞自己、也不会撞其他蛇的方向"""
    head = game.snakes[snake_idx]["body"][0]
    avoid = set()
    for i, s in enumerate(game.snakes):
        for j, p in enumerate(s["body"]):
            if i == snake_idx and j == 0:
                continue
            avoid.add(tuple(p))

    safe = []
    for d in Direction:
        dx, dy = d.value
        nx = (head[0] + dx) % game.width
        ny = (head[1] + dy) % game.height
        if (nx, ny) not in avoid:
            safe.append(d)
    return safe


def manhattan(a: list[int], b: list[int], w: int, h: int) -> int:
    dx = min(abs(a[0] - b[0]), w - abs(a[0] - b[0]))
    dy = min(abs(a[1] - b[1]), h - abs(a[1] - b[1]))
    return dx + dy


def choose_direction(
    game: SnakeGame,
    snake_idx: int,
    randomness: float = 0.18,
    mistake_rate: float = 0.15,
) -> Direction:
    """
    为第 snake_idx 条蛇选择方向
    """
    safe = get_safe_directions(game, snake_idx)
    if not safe:
        return Direction.RIGHT

    if random.random() < randomness:
        return random.choice(safe)

    s = game.snakes[snake_idx]
    head = s["body"][0]
    targets = []
    if s["x2"] and not s["x2_active"]:
        if random.random() >= mistake_rate:
            targets.append(s["x2"])
    targets.append(s["food"])

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

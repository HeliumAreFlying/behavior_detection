"""
贪吃蛇游戏核心逻辑（支持多蛇）
- 蛇撞自己或撞其他蛇即结束，标注错误
- 每蛇独立 food/x2，颜色与蛇身同色系、深浅区分
- 50%概率生成x2，先吃x2再吃食物可得2分，否则1分
"""

import random
from typing import Optional
from enum import Enum
from dataclasses import dataclass, field


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


# 每阵营一种颜色（蛇身/食物/x2 同色，用 H/F/x2 文字区分）
SNAKE_COLORS = [
    {"body": (76, 175, 80), "food": (76, 175, 80), "x2": (76, 175, 80)},    # 绿
    {"body": (33, 150, 243), "food": (33, 150, 243), "x2": (33, 150, 243)},  # 蓝
    {"body": (255, 152, 0), "food": (255, 152, 0), "x2": (255, 152, 0)},     # 橙
]


@dataclass
class SnakeState:
    body: list[list[int]]
    food: list[int]
    x2: Optional[list[int]]
    score: int
    x2_active: bool
    color_id: int

    def to_dict(self) -> dict:
        return {
            "body": [p.copy() for p in self.body],
            "food": self.food.copy(),
            "x2": self.x2.copy() if self.x2 else None,
            "score": self.score,
            "x2_active": self.x2_active,
            "color_id": self.color_id,
        }


@dataclass
class GameState:
    """游戏状态快照"""
    snakes: list[SnakeState]
    step: int

    def to_dict(self) -> dict:
        return {
            "snakes": [s.to_dict() for s in self.snakes],
            "step": self.step,
        }


class SnakeGame:
    def __init__(
        self,
        width: int = 15,
        height: int = 15,
        num_snakes: int = 1,
        seed: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.num_snakes = num_snakes
        self.rng = random.Random(seed)

        # 每条蛇: body, food, x2, score, x2_active
        self.snakes: list[dict] = []
        occupied = set()

        positions = [
            (width // 2, height // 2),
            (width // 4, height // 2),
            (3 * width // 4, height // 2),
        ]
        for i in range(num_snakes):
            pos = positions[i] if i < len(positions) else self._random_empty_pos(occupied)
            body = [[pos[0], pos[1]]]
            occupied.add((pos[0], pos[1]))
            self.snakes.append({
                "body": body,
                "food": [0, 0],
                "x2": None,
                "score": 0,
                "x2_active": False,
                "color_id": i % len(SNAKE_COLORS),
            })

        self.step_count = 0
        for i in range(num_snakes):
            self._spawn_food_for(i)

    def _all_occupied(self) -> set[tuple[int, int]]:
        out = set()
        for s in self.snakes:
            out.update(tuple(p) for p in s["body"])
            out.add(tuple(s["food"]))
            if s["x2"]:
                out.add(tuple(s["x2"]))
        return out

    def _spawn_food_for(self, snake_idx: int) -> None:
        s = self.snakes[snake_idx]
        occupied = self._all_occupied()
        occupied.discard(tuple(s["food"]))
        occupied.discard(tuple(s["x2"]) if s["x2"] else (-1, -1))

        s["food"] = self._random_empty_pos(occupied)
        occupied.add(tuple(s["food"]))
        s["x2"] = None
        if self.rng.random() < 0.5:
            pos = self._random_empty_pos(occupied)
            if pos:
                s["x2"] = pos

    def respawn_food_for(self, snake_idx: int) -> None:
        self._spawn_food_for(snake_idx)

    def _random_empty_pos(self, occupied: set) -> list[int]:
        candidates = [
            [x, y] for x in range(self.width) for y in range(self.height)
            if (x, y) not in occupied
        ]
        if not candidates:
            return [0, 0]
        return self.rng.choice(candidates)

    def get_state(self) -> GameState:
        return GameState(
            snakes=[
                SnakeState(
                    body=[p.copy() for p in s["body"]],
                    food=s["food"].copy(),
                    x2=s["x2"].copy() if s["x2"] else None,
                    score=s["score"],
                    x2_active=s["x2_active"],
                    color_id=s["color_id"],
                )
                for s in self.snakes
            ],
            step=self.step_count,
        )

    def move_all(self, directions: list[Direction]) -> tuple[bool, str, Optional[int] | tuple[int, ...]]:
        """
        所有蛇同时移动一步
        返回: (是否继续, 事件, 触发的蛇索引或None，蛇间碰撞时为 (i,j) 双方索引)
        事件: 'continue' | 'ate_food' | 'ate_food_x2' | 'ate_x2' | 'self_collision' | 'snake_collision'
        """
        if len(directions) != self.num_snakes:
            return False, "self_collision", 0

        new_heads = []
        new_bodies = []
        for i, d in enumerate(directions):
            head = self.snakes[i]["body"][0]
            dx, dy = d.value
            nh = [(head[0] + dx) % self.width, (head[1] + dy) % self.height]
            new_heads.append(nh)
            new_bodies.append([nh] + self.snakes[i]["body"][:-1])

        # 检查蛇间碰撞、撞自己
        for i in range(self.num_snakes):
            nh = tuple(new_heads[i])
            for j in range(self.num_snakes):
                if i == j:
                    if nh in set(tuple(p) for p in new_bodies[i][1:]):
                        return False, "self_collision", i
                else:
                    if nh in set(tuple(p) for p in new_bodies[j]):
                        return False, "snake_collision", (i, j)  # 双方都错
            for j in range(i + 1, self.num_snakes):
                if new_heads[i] == new_heads[j]:
                    return False, "snake_collision", (i, j)  # 头对头，双方都错

        self.step_count += 1
        eating_snake: Optional[int] = None
        event = "continue"

        for i in range(self.num_snakes):
            s = self.snakes[i]
            nh = new_heads[i]

            if nh == s["food"]:
                used = s["x2_active"]
                s["score"] += 2 if used else 1
                s["x2_active"] = False
                self._spawn_food_for(i)
                eating_snake = i
                event = "ate_food_x2" if used else "ate_food"
                break
            if s["x2"] and nh == s["x2"]:
                s["x2_active"] = True
                s["x2"] = None
                eating_snake = i
                event = "ate_x2"
                break

        for i in range(self.num_snakes):
            body = self.snakes[i]["body"]
            nh = new_heads[i]
            if eating_snake is not None and i == eating_snake and event in ("ate_food", "ate_food_x2"):
                self.snakes[i]["body"] = [nh] + body
            else:
                self.snakes[i]["body"] = [nh] + body[:-1]

        return True, event if eating_snake is not None else "continue", eating_snake

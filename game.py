"""
贪吃蛇游戏核心逻辑
- 无墙壁，蛇撞自己即结束
- 吃1个食物+1格长度
- 50%概率生成x2，先吃x2再吃食物可得2分，否则1分
- x2每波食物只生效一次，生成新食物时x2失效
"""

import random
from typing import Optional
from enum import Enum
from dataclasses import dataclass


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


@dataclass
class GameState:
    """游戏状态快照，用于场景记录"""
    snake: list[list[int]]  # [[x,y], ...] 蛇身坐标
    food: list[int]         # [x, y]
    x2: Optional[list[int]] # [x, y] 或 None
    score: int
    x2_active: bool         # 蛇是否已吃x2（本波有效）
    step: int

    def to_dict(self) -> dict:
        return {
            "snake": self.snake.copy(),
            "food": self.food.copy(),
            "x2": self.x2.copy() if self.x2 else None,
            "score": self.score,
            "x2_active": self.x2_active,
            "step": self.step,
        }


class SnakeGame:
    def __init__(self, width: int = 15, height: int = 15, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

        # 蛇：头部在前
        cx, cy = width // 2, height // 2
        self.snake: list[list[int]] = [[cx, cy]]
        self.direction = Direction.RIGHT

        self.food: list[int] = [0, 0]
        self.x2: Optional[list[int]] = None
        self.score = 0
        self.x2_active = False  # 本波是否已吃x2
        self.step_count = 0

        self._spawn_food()

    def _spawn_food(self) -> None:
        """生成新食物，50%概率同时生成x2"""
        occupied = set(tuple(p) for p in self.snake)
        self.food = self._random_empty_pos(occupied)
        occupied.add(tuple(self.food))

        # x2 每波重置：生成新食物时上一波x2失效
        self.x2 = None
        if self.rng.random() < 0.5:
            pos = self._random_empty_pos(occupied)
            if pos is not None:
                self.x2 = pos

    def _random_empty_pos(self, occupied: set) -> list[int]:
        """在空位中随机选一个"""
        candidates = [
            [x, y] for x in range(self.width) for y in range(self.height)
            if (x, y) not in occupied
        ]
        if not candidates:
            return [0, 0]
        return self.rng.choice(candidates)

    def get_state(self) -> GameState:
        return GameState(
            snake=[p.copy() for p in self.snake],
            food=self.food.copy(),
            x2=self.x2.copy() if self.x2 else None,
            score=self.score,
            x2_active=self.x2_active,
            step=self.step_count,
        )

    def move(self, direction: Direction) -> tuple[bool, str]:
        """
        执行一步移动
        返回: (是否继续游戏, 本步事件描述)
        事件: 'continue' | 'ate_food' | 'ate_food_x2' | 'ate_x2' | 'collision'
        """
        dx, dy = direction.value
        head = self.snake[0]
        new_head = [head[0] + dx, head[1] + dy]

        # 无墙壁：环形穿越
        new_head[0] = new_head[0] % self.width
        new_head[1] = new_head[1] % self.height

        # 撞自己
        body_set = set(tuple(p) for p in self.snake)
        if tuple(new_head) in body_set:
            return False, "collision"

        self.step_count += 1
        self.snake.insert(0, new_head)

        # 吃食物（苹果）
        if new_head == self.food:
            used_x2 = self.x2_active
            self.score += 2 if used_x2 else 1
            self.x2_active = False
            self._spawn_food()
            return True, "ate_food_x2" if used_x2 else "ate_food"

        # 吃x2（不增长，只激活加成）
        if self.x2 and new_head == self.x2:
            self.x2_active = True
            self.x2 = None
            self.snake.pop()
            return True, "ate_x2"

        self.snake.pop()
        return True, "continue"

"""
演示 UI：读取 dataset.json 回放贪吃蛇场景
"""

import json
import sys
from pathlib import Path

try:
    import pygame
except ImportError:
    print("请先安装 pygame: pip install pygame")
    sys.exit(1)

try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None


# 颜色
BG = (28, 28, 32)
GRID_LINE = (48, 48, 56)
SNAKE_HEAD = (76, 175, 80)
SNAKE_BODY = (56, 142, 60)
FOOD = (244, 67, 54)
X2 = (255, 193, 7)
TEXT = (220, 220, 220)
TEXT_DIM = (140, 140, 140)
CORRECT = (76, 175, 80)
INCORRECT = (244, 67, 54)
BTN_BG = (60, 60, 68)
BTN_HOVER = (78, 78, 88)

# 标注原因中文说明（用括号避免 ✓✗ 等符号在部分字体下乱码）
REASON_NAMES = {
    "ate_x2_then_food": "先吃x2再吃食物 (对)",
    "ate_food_no_x2": "无x2时吃食物 (对)",
    "self_collision": "撞到自己 (错)",
    "x2_wasted": "先吃食物导致x2浪费 (错)",
    "timeout": "超时未吃食物 (错)",
}


def load_dataset(path: str | Path) -> tuple[list[dict], Path | None]:
    """返回 (episodes, path)，失败时 episodes 为空"""
    p = Path(path)
    if not p.exists():
        return [], None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        episodes = data.get("episodes", [])
        return episodes, p
    except Exception:
        return [], None


def choose_json_file(initial_dir: Path | None = None) -> Path | None:
    """打开文件选择对话框，返回选中的 JSON 路径"""
    if filedialog is None:
        return None
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    initial = str(initial_dir or Path.cwd())
    path = filedialog.askopenfilename(
        title="选择回放数据文件",
        initialdir=initial,
        filetypes=[("JSON 数据", "*.json"), ("所有文件", "*.*")],
    )
    root.destroy()
    return Path(path) if path else None


def draw_scene(
    screen: pygame.Surface,
    scene: dict,
    cell_size: int,
    margin_x: int,
    margin_y: int,
    grid_w: int,
    grid_h: int,
    panel_x: int,
    panel_w: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    label: str,
    reason: str,
    episode_idx: int,
    total_episodes: int,
    scene_idx: int,
    total_scenes: int,
    current_path: str | None,
    open_btn_rect: pygame.Rect,
    btn_hover: bool,
) -> None:
    screen.fill(BG)

    snake = scene.get("snake", [])
    food = scene.get("food", [0, 0])
    x2 = scene.get("x2")
    score = scene.get("score", 0)
    step = scene.get("step", scene_idx)
    grid_y = margin_y

    # 左侧：纯游戏网格，无任何覆盖
    # 绘制网格
    for x in range(grid_w + 1):
        pygame.draw.line(
            screen, GRID_LINE,
            (margin_x + x * cell_size, grid_y),
            (margin_x + x * cell_size, grid_y + grid_h * cell_size),
        )
    for y in range(grid_h + 1):
        pygame.draw.line(
            screen, GRID_LINE,
            (margin_x, grid_y + y * cell_size),
            (margin_x + grid_w * cell_size, grid_y + y * cell_size),
        )

    # 食物
    if food:
        fx, fy = food[0] % grid_w, food[1] % grid_h
        rect = (margin_x + fx * cell_size + 2, grid_y + fy * cell_size + 2,
                cell_size - 4, cell_size - 4)
        pygame.draw.rect(screen, FOOD, rect, border_radius=4)

    # x2
    if x2:
        xx, xy = x2[0] % grid_w, x2[1] % grid_h
        rect = (margin_x + xx * cell_size + 2, grid_y + xy * cell_size + 2,
                cell_size - 4, cell_size - 4)
        pygame.draw.rect(screen, X2, rect, border_radius=4)
        txt = small_font.render("x2", True, (40, 40, 40))
        tw, th = txt.get_size()
        screen.blit(txt, (margin_x + xx * cell_size + (cell_size - tw) // 2,
                         grid_y + xy * cell_size + (cell_size - th) // 2))

    # 蛇身
    for i, (sx, sy) in enumerate(snake):
        sx, sy = sx % grid_w, sy % grid_h
        rect = (margin_x + sx * cell_size + 1, grid_y + sy * cell_size + 1,
                cell_size - 2, cell_size - 2)
        color = SNAKE_HEAD if i == 0 else SNAKE_BODY
        pygame.draw.rect(screen, color, rect, border_radius=3)

    # 无数据时显示提示（网格中央）
    if total_episodes == 0:
        hint = small_font.render("请点击「打开文件」或按 O 键选择 JSON 数据文件", True, TEXT_DIM)
        hw, hh = hint.get_size()
        center_x = margin_x + (grid_w * cell_size - hw) // 2
        center_y = grid_y + (grid_h * cell_size - hh) // 2
        screen.blit(hint, (center_x, center_y))

    # 右侧信息面板：一行行排列
    reason_zh = REASON_NAMES.get(reason, reason) if total_episodes > 0 else "-"
    line_h = 28
    y = margin_y

    # 1. 打开文件按钮
    btn_color = BTN_HOVER if btn_hover else BTN_BG
    pygame.draw.rect(screen, btn_color, open_btn_rect, border_radius=4)
    btn_txt = small_font.render("打开文件 (O)", True, TEXT)
    bt_w, bt_h = btn_txt.get_size()
    screen.blit(btn_txt, (open_btn_rect.x + (open_btn_rect.w - bt_w) // 2,
                         open_btn_rect.y + (open_btn_rect.h - bt_h) // 2))
    y += open_btn_rect.h + line_h

    # 2. 文件路径（可换行）
    if current_path:
        path_str = str(current_path)
        if len(path_str) <= 32:
            path_txt = small_font.render(path_str, True, TEXT_DIM)
            screen.blit(path_txt, (panel_x, y))
        else:
            path_txt = small_font.render(path_str[-32:], True, TEXT_DIM)
            screen.blit(path_txt, (panel_x, y))
        y += line_h - 4
    y += line_h

    # 3. 标注（正确/错误 + 原因）
    if total_episodes > 0 and label not in ("-", ""):
        label_zh = "正确 (对)" if label == "correct" else "错误 (错)"
        label_color = CORRECT if label == "correct" else INCORRECT
        label_txt = font.render(label_zh, True, label_color)
        screen.blit(label_txt, (panel_x, y))
        y += line_h
        reason_txt = small_font.render(reason_zh, True, TEXT)
        screen.blit(reason_txt, (panel_x, y))
        y += line_h
    else:
        y += line_h * 2

    # 4. 局数 / 帧
    info_txt = font.render(f"局数: {episode_idx + 1}/{total_episodes}", True, TEXT)
    screen.blit(info_txt, (panel_x, y))
    y += line_h
    info_txt = font.render(f"帧: {scene_idx + 1}/{total_scenes}", True, TEXT)
    screen.blit(info_txt, (panel_x, y))
    y += line_h

    # 5. 步数 / 得分
    info_txt = font.render(f"步数: {step}  得分: {score}", True, TEXT)
    screen.blit(info_txt, (panel_x, y))
    y += line_h

    # 6. 行为标注
    anno_color = CORRECT if label == "correct" else (INCORRECT if label == "incorrect" else TEXT)
    anno_txt = small_font.render(f"行为标注: {reason_zh}", True, anno_color)
    screen.blit(anno_txt, (panel_x, y))
    y += line_h + 8

    # 7. 控制说明
    ctrl_txt = small_font.render("空格:播放/暂停", True, TEXT_DIM)
    screen.blit(ctrl_txt, (panel_x, y))
    y += line_h - 4
    ctrl_txt = small_font.render("←→:帧  A/D:局", True, TEXT_DIM)
    screen.blit(ctrl_txt, (panel_x, y))
    y += line_h - 4
    ctrl_txt = small_font.render("Home/End:首尾", True, TEXT_DIM)
    screen.blit(ctrl_txt, (panel_x, y))


def main():
    pygame.init()
    base = Path(__file__).parent
    if len(sys.argv) > 1:
        default_path = Path(sys.argv[1])
    elif (base / "dataset.json").exists():
        default_path = base / "dataset.json"
    else:
        default_path = base / "batches" / "batch_00000.json"
    episodes, current_path = load_dataset(default_path)

    cell_size = 40
    grid_w = 15
    grid_h = 15
    margin_x = 32
    margin_y = 24
    gap = 32
    panel_w = 260
    grid_size = grid_w * cell_size
    width = margin_x + grid_size + gap + panel_w + margin_x
    height = margin_y * 2 + grid_h * cell_size

    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    pygame.display.set_caption("贪吃蛇 回放演示")

    font = pygame.font.SysFont("microsoftyahei,simhei,arial", 20)
    small_font = pygame.font.SysFont("microsoftyahei,simhei,arial", 15)

    panel_x = margin_x + grid_size + gap
    open_btn_rect = pygame.Rect(panel_x, margin_y, 140, 36)
    last_file_dir = current_path.parent if current_path else Path(__file__).parent

    episode_idx = 0
    scene_idx = 0
    playing = False
    fps = 8
    clock = pygame.time.Clock()
    frame_counter = 0

    def try_load(path: Path) -> bool:
        nonlocal episodes, current_path, episode_idx, scene_idx, last_file_dir
        eps, p = load_dataset(path)
        if eps:
            episodes = eps
            current_path = p
            episode_idx = 0
            scene_idx = 0
            last_file_dir = path.parent
            return True
        return False

    while True:
        mouse_pos = pygame.mouse.get_pos()
        btn_hover = open_btn_rect.collidepoint(mouse_pos)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if open_btn_rect.collidepoint(mouse_pos) and filedialog:
                    new_path = choose_json_file(last_file_dir)
                    if new_path:
                        try_load(new_path)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_o:
                    if filedialog:
                        new_path = choose_json_file(last_file_dir)
                        if new_path:
                            try_load(new_path)
                elif not episodes:
                    continue
                elif e.key == pygame.K_SPACE:
                    playing = not playing
                elif e.key == pygame.K_LEFT:
                    playing = False
                    scene_idx = max(0, scene_idx - 1)
                elif e.key == pygame.K_RIGHT:
                    playing = False
                    ep = episodes[episode_idx]
                    scenes = ep.get("scenes", [])
                    scene_idx = min(len(scenes) - 1, scene_idx + 1)
                elif e.key == pygame.K_a:
                    episode_idx = (episode_idx - 1) % len(episodes)
                    scene_idx = 0
                    playing = False
                elif e.key == pygame.K_d:
                    episode_idx = (episode_idx + 1) % len(episodes)
                    scene_idx = 0
                    playing = False
                elif e.key == pygame.K_HOME:
                    scene_idx = 0
                elif e.key == pygame.K_END:
                    ep = episodes[episode_idx]
                    scenes = ep.get("scenes", [])
                    scene_idx = len(scenes) - 1
                    playing = False

        if not episodes:
            scene = {"snake": [], "food": [0, 0], "x2": None, "score": 0, "x2_active": False, "step": 0}
            ep = {"label": "-", "reason": "-"}
            path_str = str(current_path) if current_path else None
        else:
            ep = episodes[episode_idx]
            scenes = ep.get("scenes", [])
            if not scenes:
                scene_idx = 0
                playing = False
                scene = {"snake": [], "food": [0, 0], "x2": None, "score": 0, "x2_active": False, "step": 0}
            else:
                scene_idx = min(scene_idx, len(scenes) - 1)
                scene = scenes[scene_idx]
                if playing:
                    frame_counter += 1
                    if frame_counter >= 60 // fps:
                        frame_counter = 0
                        if scene_idx < len(scenes) - 1:
                            scene_idx += 1
                        else:
                            playing = False
            path_str = str(current_path) if current_path else None

        draw_scene(
            screen, scene, cell_size, margin_x, margin_y,
            grid_w, grid_h, panel_x, panel_w,
            font, small_font,
            ep.get("label", "-"), ep.get("reason", "-"),
            episode_idx, len(episodes) or 1,
            scene_idx, len(ep.get("scenes", [])) or 1,
            path_str, open_btn_rect, btn_hover,
        )

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()

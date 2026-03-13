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


from game import SNAKE_COLORS

# 颜色
BG = (28, 28, 32)
GRID_LINE = (48, 48, 56)
TEXT = (220, 220, 220)
TEXT_DIM = (140, 140, 140)
CORRECT = (76, 175, 80)
INCORRECT = (244, 67, 54)
BTN_BG = (60, 60, 68)
BTN_HOVER = (78, 78, 88)

# 标注原因中文说明（用括号避免 ✓✗ 等符号在部分字体下乱码）
REASON_NAMES = {
    "ate_x2_then_food": "先吃x2再吃食物 (对)",
    "ate_food_no_x2": "吃食物 (对)",
    "in_progress": "进行中",
    "self_collision": "撞到自己 (错)",
    "snake_collision": "蛇间碰撞 (错)",
    "x2_wasted": "先吃食物导致x2浪费 (错)",
    "timeout": "超时未吃食物 (错)",
}


def _get_total_score(scene: dict) -> int:
    """兼容新格式 snakes / 旧格式 snake+score"""
    if "snakes" in scene:
        return sum(s.get("score", 0) for s in scene["snakes"])
    return scene.get("score", 0)


def _any_snake_had_x2(scene: dict) -> bool:
    """兼容新格式 snakes / 旧格式 x2"""
    if "snakes" in scene:
        return any(s.get("x2") is not None for s in scene["snakes"])
    return scene.get("x2") is not None


def _get_food(scene: dict, snake_idx: int) -> list[int] | None:
    """获取指定蛇的食物坐标"""
    if "snakes" in scene and snake_idx < len(scene["snakes"]):
        return scene["snakes"][snake_idx].get("food")
    if snake_idx == 0:
        return scene.get("food")
    return None


def infer_annotation_so_far(
    scenes: list[dict], up_to_idx: int, episode_label: str, episode_reason: str
) -> tuple[str, str]:
    """
    推断单蛇「当前食物」的标注。只针对当前这波食物，不用首/末标注。
    """
    if not scenes or up_to_idx < 0:
        return "correct", "in_progress"
    n = len(scenes)
    up_to_idx = min(up_to_idx, n - 1)
    if up_to_idx == n - 1 and episode_reason in ("self_collision", "snake_collision", "timeout"):
        return episode_label, episode_reason

    # 找到当前食物这波的起点（食物位置变化处）
    wave_start = 0
    for k in range(1, up_to_idx + 1):
        if _get_food(scenes[k - 1], 0) != _get_food(scenes[k], 0):
            wave_start = k

    x2_wasted = False
    used_x2_ok = False
    any_eat = False
    for k in range(wave_start, up_to_idx):
        a, b = scenes[k], scenes[k + 1]
        score_a = _get_total_score(a)
        score_b = _get_total_score(b)
        ds = score_b - score_a
        had_x2 = _any_snake_had_x2(a)
        if ds == 1 and had_x2:
            x2_wasted = True
            any_eat = True
        elif ds == 2:
            used_x2_ok = True
            any_eat = True
        elif ds == 1:
            any_eat = True

    if x2_wasted:
        return "incorrect", "x2_wasted"
    if used_x2_ok:
        return "correct", "ate_x2_then_food"
    if any_eat:
        return "correct", "ate_food_no_x2"
    return "correct", "in_progress"


def infer_snake_annotations_so_far(
    scenes: list[dict], up_to_idx: int, final_snake_annotations: list[dict]
) -> list[dict]:
    """
    推断每条蛇「当前食物」的标注。只针对当前这波食物，不用首/末标注。
    """
    if not scenes or up_to_idx < 0 or not final_snake_annotations:
        return final_snake_annotations
    n = len(scenes)
    up_to_idx = min(up_to_idx, n - 1)
    num_snakes = len(final_snake_annotations)

    # 最后一帧且是碰撞/超时：无法从得分推断，用存储的标注
    unreachable_reasons = ("self_collision", "snake_collision", "timeout")
    if up_to_idx == n - 1 and any(
        a.get("reason") in unreachable_reasons for a in final_snake_annotations
    ):
        return final_snake_annotations

    first_scene = scenes[0]
    if "snakes" not in first_scene or len(first_scene["snakes"]) < num_snakes:
        lbl, rsn = infer_annotation_so_far(
            scenes, up_to_idx,
            final_snake_annotations[0].get("label", "correct"),
            final_snake_annotations[0].get("reason", "ate_food_no_x2"),
        )
        return [{"label": lbl, "reason": rsn}]

    result: list[dict] = []
    for i in range(num_snakes):
        # 找到当前食物这波的起点
        wave_start = 0
        for k in range(1, up_to_idx + 1):
            if _get_food(scenes[k - 1], i) != _get_food(scenes[k], i):
                wave_start = k

        first_err: str | None = None
        last_ok = "in_progress"
        for k in range(wave_start, up_to_idx):
            a, b = scenes[k], scenes[k + 1]
            sa = a["snakes"][i] if i < len(a.get("snakes", [])) else {}
            sb = b["snakes"][i] if i < len(b.get("snakes", [])) else {}
            score_a = sa.get("score", 0)
            score_b = sb.get("score", 0)
            delta = score_b - score_a
            had_x2 = sa.get("x2") is not None
            if delta == 1 and had_x2:
                first_err = "x2_wasted"
            elif delta == 2:
                last_ok = "ate_x2_then_food"
            elif delta == 1:
                last_ok = "ate_food_no_x2"
        result.append({
            "label": "incorrect" if first_err else "correct",
            "reason": first_err or last_ok,
        })
    return result


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


def _draw_snake_head(screen: pygame.Surface, cx: int, cy: int, r: int, col: tuple,
                     shape: str, dx: int = 1, dy: int = 0) -> None:
    """蛇头形状：diamond 开局 | triangle 前进 | circle 撞击死亡"""
    if shape == "diamond":
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        pygame.draw.polygon(screen, col, pts)
        pygame.draw.polygon(screen, (40, 40, 40), pts, 1)
    elif shape == "circle":
        pygame.draw.circle(screen, col, (cx, cy), r)
        pygame.draw.circle(screen, (40, 40, 40), (cx, cy), r, 1)
    else:
        if dx == 0 and dy == 0:
            dx, dy = 1, 0
        tip = (cx + dx * r, cy + dy * r)
        perp = (-dy, dx)
        base1 = (cx - dx * r * 0.6 + perp[0] * r * 0.6, cy - dy * r * 0.6 + perp[1] * r * 0.6)
        base2 = (cx - dx * r * 0.6 - perp[0] * r * 0.6, cy - dy * r * 0.6 - perp[1] * r * 0.6)
        pts = [tip, base1, base2]
        pygame.draw.polygon(screen, col, pts)
        pygame.draw.polygon(screen, (40, 40, 40), pts, 1)


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
    snake_annotations: list[dict] | None = None,
    prev_scene: dict | None = None,
) -> None:
    screen.fill(BG)

    # 兼容新格式 snakes / 旧格式 snake
    snakes_data = scene.get("snakes")
    if snakes_data is None:
        snakes_data = [{
            "body": scene.get("snake", []),
            "food": scene.get("food", [0, 0]),
            "x2": scene.get("x2"),
            "score": scene.get("score", 0),
            "color_id": 0,
        }]
    total_score = sum(s.get("score", 0) for s in snakes_data)
    step = scene.get("step", scene_idx)
    grid_y = margin_y

    # 左侧：纯游戏网格，无任何覆盖
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

    # 食物(F)、x2、蛇身(H/无字)，每阵营一色
    for s in snakes_data:
        cid = s.get("color_id", 0) % len(SNAKE_COLORS)
        col = SNAKE_COLORS[cid]["body"]
        food = s.get("food", [0, 0])
        x2 = s.get("x2")
        if food:
            fx, fy = food[0] % grid_w, food[1] % grid_h
            rect = (margin_x + fx * cell_size + 2, grid_y + fy * cell_size + 2,
                    cell_size - 4, cell_size - 4)
            pygame.draw.rect(screen, col, rect, border_radius=4)
            txt = small_font.render("F", True, (40, 40, 40))
            tw, th = txt.get_size()
            screen.blit(txt, (margin_x + fx * cell_size + (cell_size - tw) // 2,
                             grid_y + fy * cell_size + (cell_size - th) // 2))
        if x2:
            xx, xy = x2[0] % grid_w, x2[1] % grid_h
            rect = (margin_x + xx * cell_size + 2, grid_y + xy * cell_size + 2,
                    cell_size - 4, cell_size - 4)
            pygame.draw.rect(screen, col, rect, border_radius=4)
            txt = small_font.render("x2", True, (40, 40, 40))
            tw, th = txt.get_size()
            screen.blit(txt, (margin_x + xx * cell_size + (cell_size - tw) // 2,
                             grid_y + xy * cell_size + (cell_size - th) // 2))

    # 蛇身：蛇头按状态画菱形/三角/圆
    anns = scene.get("snake_annotations", [])
    is_last = (scene_idx == total_scenes - 1) if total_scenes else False
    dead_reasons = {"self_collision", "snake_collision"}
    prev_heads = []
    if prev_scene and prev_scene.get("snakes"):
        for sp in prev_scene["snakes"]:
            b = sp.get("body", [])
            prev_heads.append((int(b[0][0]) % grid_w, int(b[0][1]) % grid_h) if b else None)

    for si, s in enumerate(snakes_data):
        body = s.get("body", [])
        cid = s.get("color_id", 0) % len(SNAKE_COLORS)
        col = SNAKE_COLORS[cid]["body"]
        for i, pos in enumerate(body):
            try:
                sx, sy = int(pos[0]), int(pos[1])
            except (IndexError, TypeError):
                continue
            sx, sy = sx % grid_w, sy % grid_h
            rect = (margin_x + sx * cell_size + 1, grid_y + sy * cell_size + 1,
                    cell_size - 2, cell_size - 2)
            pygame.draw.rect(screen, col, rect, border_radius=3)
            if i == 0:
                cx = margin_x + sx * cell_size + cell_size // 2
                cy = grid_y + sy * cell_size + cell_size // 2
                r = cell_size // 2 - 2
                ann = anns[si] if si < len(anns) else {}
                reason = ann.get("reason", "")
                if scene_idx == 0:
                    _draw_snake_head(screen, cx, cy, r, col, "diamond")
                elif is_last and reason in dead_reasons:
                    _draw_snake_head(screen, cx, cy, r, col, "circle")
                else:
                    dx, dy = 0, 0
                    if len(body) >= 2:
                        dx = body[0][0] - body[1][0]
                        dy = body[0][1] - body[1][1]
                    elif si < len(prev_heads) and prev_heads[si]:
                        ph = prev_heads[si]
                        dx = sx - ph[0]
                        dy = sy - ph[1]
                    if dx > 1: dx = -1
                    elif dx < -1: dx = 1
                    if dy > 1: dy = -1
                    elif dy < -1: dy = 1
                    if dx == 0 and dy == 0:
                        dx = 1
                    _draw_snake_head(screen, cx, cy, r, col, "triangle", dx, dy)

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

    # 3. 标注（每蛇单独 / 或单蛇汇总）
    if total_episodes > 0 and (label not in ("-", "") or snake_annotations):
        if snake_annotations:
            for i, ann in enumerate(snake_annotations):
                l, r = ann.get("label", "correct"), ann.get("reason", "")
                lbl_zh = "正确" if l == "correct" else "错误"
                rsn_zh = REASON_NAMES.get(r, r)
                snake_color = SNAKE_COLORS[i % len(SNAKE_COLORS)]["body"]
                line_color = snake_color if l == "correct" else INCORRECT
                txt = small_font.render(f"蛇{i+1}: {lbl_zh} - {rsn_zh}", True, line_color)
                screen.blit(txt, (panel_x, y))
                y += line_h - 2
            y += 4
        else:
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
    info_txt = font.render(f"步数: {step}  得分: {total_score}", True, TEXT)
    screen.blit(info_txt, (panel_x, y))
    y += line_h + 8

    # 6. 控制说明
    ctrl_txt = small_font.render("空格:播放/暂停", True, TEXT_DIM)
    screen.blit(ctrl_txt, (panel_x, y))
    y += line_h - 4
    ctrl_txt = small_font.render("←→:帧  A/D或PgUp/Dn:局", True, TEXT_DIM)
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

    cell_size = 43  # 15*43≈645，接近 640x640
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

    # 禁用 IME 文本输入，避免 A/D 等字母键被输入法拦截导致失效
    pygame.key.stop_text_input()
    # 关闭按键连发，避免长按导致多次触发
    pygame.key.set_repeat()

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
                k, sc = e.key, getattr(e, "scancode", 0)
                # 用 key 或 scancode 判断，避免 IME 下 key 异常（A=30,D=32,Left=75,Right=77,O=24）
                def hit(key_val, scan):
                    return k == key_val or sc == scan
                if hit(pygame.K_o, 24):
                    if filedialog:
                        new_path = choose_json_file(last_file_dir)
                        if new_path:
                            try_load(new_path)
                elif not episodes:
                    continue
                elif hit(pygame.K_SPACE, 57):
                    playing = not playing
                elif hit(pygame.K_LEFT, 75):
                    playing = False
                    scene_idx = max(0, scene_idx - 1)
                elif hit(pygame.K_RIGHT, 77):
                    playing = False
                    ep = episodes[episode_idx]
                    scenes = ep.get("scenes", [])
                    scene_idx = min(len(scenes) - 1, scene_idx + 1)
                elif hit(pygame.K_a, 30) or hit(pygame.K_PAGEUP, 73):
                    episode_idx = (episode_idx - 1) % len(episodes)
                    scene_idx = 0
                    playing = False
                elif hit(pygame.K_d, 32) or hit(pygame.K_PAGEDOWN, 81):
                    episode_idx = (episode_idx + 1) % len(episodes)
                    scene_idx = 0
                    playing = False
                elif hit(pygame.K_HOME, 71):
                    scene_idx = 0
                elif hit(pygame.K_END, 79):
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

        # 优先使用预计算的每帧标注（生成时已写入 scene）；无则回退到推断
        if episodes and ep.get("scenes"):
            if "snake_annotations" in scene:
                snake_ann = scene["snake_annotations"]
                disp_label = snake_ann[0]["label"] if snake_ann else "correct"
                disp_reason = snake_ann[0]["reason"] if snake_ann else "in_progress"
            else:
                disp_label, disp_reason = infer_annotation_so_far(
                    ep["scenes"], scene_idx,
                    ep.get("label", "correct"), ep.get("reason", "ate_food_no_x2"),
                )
                snake_ann = infer_snake_annotations_so_far(
                    ep["scenes"], scene_idx,
                    ep.get("snake_annotations") or [{"label": disp_label, "reason": disp_reason}],
                )
        else:
            disp_label, disp_reason = ep.get("label", "-"), ep.get("reason", "-")
            snake_ann = None
        scenes = ep.get("scenes", []) if episodes else []
        prev_sc = scenes[scene_idx - 1] if scene_idx > 0 and scenes else None
        draw_scene(
            screen, scene, cell_size, margin_x, margin_y,
            grid_w, grid_h, panel_x, panel_w,
            font, small_font,
            disp_label, disp_reason,
            episode_idx, len(episodes) or 1,
            scene_idx, len(scenes) or 1,
            path_str, open_btn_rect, btn_hover,
            snake_annotations=snake_ann,
            prev_scene=prev_sc,
        )

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()

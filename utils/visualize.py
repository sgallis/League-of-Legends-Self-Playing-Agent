import cv2
import keyboard

from utils.template import read_template, minimap_template_matching,\
    load_template, gold_template_matching,\
    time_template_matching, level_template_matching, health_estimate
from utils.variables import mid_box, big_box

def view_health(game):
    while True:
        health_img = game.frame.capture_health()
        health = health_estimate(health_img)
        print("dead" if not health else health)
        # cv2.imshow("health", health_img)
        cv2.waitKey(200)

def view_level(game):
    level_template = load_template("assets/level_digits")
    while True:
        level_img = game.frame.capture_level()
        level = level_template_matching(level_img, level_template, min_score=0.75)
        print(level)
        # cv2.imshow("level", level_img)
        cv2.waitKey(200)

def view_time(game):
    time_template = load_template("assets/time_digits")
    while True:
        time_img = game.frame.capture_time()
        game_time = time_template_matching(time_img, time_template, min_score=0.7)
        print(f"{game_time}s")
        # cv2.imshow("time", time_img)
        cv2.waitKey(200)

def view_gold(game):
    gold_template = load_template()
    while True:
        gold_img = game.frame.capture_gold()
        gold = gold_template_matching(gold_img, gold_template, min_score=0.7)
        print(gold)
        cv2.imshow("gold", gold_img)
        # print(ocr_digits(gold))
        cv2.waitKey(200)

def view_minimap(game, template_path):
    template, t_w, t_h = read_template(template_path)
    while True:
        minimap = game.frame.capture_minimap() # RGB ARRAY
        _, inside, inside_big, x_c, y_c = minimap_template_matching(
            minimap,
            template,
            (t_w, t_h),
            mid_box,
            big_box
            )
        # print(x_c, y_c)
        cv2.drawContours(minimap, [mid_box.astype(int)], 0, (0, 255, 0), 2)
        cv2.drawContours(minimap, [big_box.astype(int)], 0, (0, 255, 0), 2)
        
        x1 = int(x_c - t_w / 2)
        y1 = int(y_c - t_h / 2)
        x2 = int(x_c + t_w / 2)
        y2 = int(y_c + t_h / 2)
        # minimap_rgb = cv2.cvtColor(minimap, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(
            minimap,
            (x1, y1),
            (x2, y2),
            (0, 255, 0) if inside else ((255, 0, 0) if inside_big else (0, 0, 255)),
            2
            )
        cv2.imshow("minimap", minimap)
        cv2.waitKey(200)

def view_game_state(game, stop_key):
    t_w = game.frame.champion_template_w
    t_h = game.frame.champion_template_h
    while True:
        game_data, images = game.get_game_data()

        x_c, y_c = game_data["activePlayer"]["position"]
        inside = game_data["activePlayer"]["inside"]
        inside_big = game_data["activePlayer"]["insideBig"]
        gold = game_data["activePlayer"]["gold"]
        level = game_data["activePlayer"]["level"]
        health = game_data["activePlayer"]["health"]
        time = game_data["game"]["time"]

        minimap = images["minimap"]

        cv2.drawContours(minimap, [mid_box.astype(int)], 0, (0, 255, 0), 2)
        cv2.drawContours(minimap, [big_box.astype(int)], 0, (0, 255, 0), 2)

        x1 = int(x_c - t_w / 2)
        y1 = int(y_c - t_h / 2)
        x2 = int(x_c + t_w / 2)
        y2 = int(y_c + t_h / 2)
        # minimap_rgb = cv2.cvtColor(minimap, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(
            minimap,
            (x1, y1),
            (x2, y2),
            (0, 255, 0) if inside else ((255, 0, 0) if inside_big else (0, 0, 255)),
            2
            )
        cv2.imshow("minimap", minimap)
        print("---------------------------------------------------------------------------------------")
        print(f"\tTime\t|\tGold\t|\tLevel\t|\tHealth\t|\tAlive")
        print(f"\t{time}s\t|\t{gold}\t|\t{level}\t|\t{health*100:.2f}%\t|\t{health > 0}")
        
        cv2.waitKey(100)
        if keyboard.is_pressed(stop_key):
            print("Stopping...")
            break

def draw_grid(img, n_h=5, n_v=10, color=(255,255,255), thickness=1, show=False, save=""):
    H, W = img.shape[:2]

    # Horizontal lines
    for i in range(1, n_h + 1):
        y = int(i * H / (n_h + 1))
        cv2.line(img, (0, y), (W, y), color, thickness)

    # Vertical lines
    for j in range(1, n_v + 1):
        x = int(j * W / (n_v + 1))
        cv2.line(img, (x, 0), (x, H), color, thickness)
    if show:
        cv2.imshow("grid", img)
    if save:
        cv2.imwrite(save, img)

def draw_game_grid(game, n_h=5, n_v=10, color=(255,255,255), thickness=1, show=False, save=""):
    img = game.capture_frame()
    draw_grid(img, n_h=n_h, n_v=n_v, color=color, thickness=thickness, show=show, save=save)

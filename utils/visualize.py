import cv2

from utils.template import read_template, template_matching
from utils.variables import mid_box, big_box

def view_template_matching(game, template_path):
    template, t_w, t_h = read_template(template_path)
    while True:
        minimap = game.capture_minimap() # GRAY ARRAY
        _, inside, inside_big, x_c, y_c = template_matching(
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

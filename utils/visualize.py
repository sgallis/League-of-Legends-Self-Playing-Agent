import cv2

from utils.template import read_template, template_matching
from utils.variables import mid_box

def view_template_matching(game, template_path):
    template, t_w, t_h = read_template(template_path)
    while True:
        minimap = game.capture_minimap() # GRAY ARRAY
        _, inside, x_c, y_c = template_matching(
            minimap,
            template,
            (t_w, t_h),
            mid_box
            )
        # cv2.drawContours(minimap, [mid_box], 0, (0, 255, 0), 2)
        x1 = int(x_c - t_w / 2)
        y1 = int(y_c - t_h / 2)
        x2 = int(x_c + t_w / 2)
        y2 = int(y_c + t_h / 2)
        minimap_rgb = cv2.cvtColor(minimap, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(
            minimap_rgb,
            (x1, y1),
            (x2, y2),
            (0, 255, 0) if inside else (0, 0, 255),
            2
            )
        cv2.imshow("minimap", minimap_rgb)
        cv2.waitKey(1)

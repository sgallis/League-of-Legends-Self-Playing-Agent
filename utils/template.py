import cv2


def read_template(template_path):
    template = cv2.imread(template_path, 0)
    t_w, t_h = template.shape[::-1]
    return template, t_w, t_h

def template_matching(minimap, template, t_wh, box, threshold=0.6):
    res = cv2.matchTemplate(minimap, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    inside = False
    above = max_val >= threshold

    top_left = max_loc  # (x, y)
    x_c, y_c = (top_left[0] + t_wh[0]//2, top_left[1] + t_wh[1]//2)
    inside = cv2.pointPolygonTest(box, (x_c, y_c), False) >= 0

    return above, inside, x_c, y_c
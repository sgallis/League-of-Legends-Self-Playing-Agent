from pathlib import Path
import cv2
import numpy as np


def load_template(folder="assets/gold_digits"):
    tmpls = {}
    for p in Path(folder).glob("*.png"):
        d = p.stem  # "0".."9"
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        tmpls[int(d)] = img
    return tmpls

def health_estimate(health_img):
    g_channel = health_img[0, :, 1]
    green = np.where(g_channel > 30)[0]
    nb_g = len(green)
    health = 0 if nb_g < 4 else nb_g / len(g_channel)
    return health

def level_template_matching(level_img, level_templates, min_score=0.75):
    level_bin = rgb2bin(level_img)
    level = int(read_digits_by_template(level_bin, level_templates, min_score=min_score, dist=0.1))
    level = 14 if level == 141 else level
    return level

def time_template_matching(time_img, time_templates, min_score=0.7):
    time_bin = rgb2bin(time_img)
    digits = read_digits_by_template(time_bin, time_templates, min_score=min_score)
    minutes, seconds = int(digits[:2]), int(digits[2:])
    return minutes * 60 + seconds

def gold_template_matching(gold_img, gold_templates, min_score=0.7):
    gold_bin = rgb2bin(gold_img)
    return int(read_digits_by_template(gold_bin, gold_templates, min_score=min_score))

def read_digits_by_template(roi_gray_bin, templates, min_score=0.45, dist=0.6):
    """
    Sliding match for each digit template; then greedy left-to-right pick.
    Works when digits are on one line and roughly same size as templates.
    """
    candidates = []
    for digit, tmpl in templates.items():
        res = cv2.matchTemplate(roi_gray_bin, tmpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= min_score)
        for (x, y) in zip(xs, ys):
            candidates.append((x, y, digit, res[y, x], tmpl.shape[1], tmpl.shape[0]))

    # sort by x then score desc
    candidates.sort(key=lambda t: (t[0], -t[3]))

    # non-max suppression in x-direction (avoid duplicates)
    picked = []
    for x, y, digit, score, w, h in candidates:
        if all(abs(x - px) > w*dist for px, *_ in picked):
            picked.append((x, digit, score))

    picked.sort(key=lambda t: t[0])
    if not picked:
        return None

    return "".join(str(d) for _, d, _ in picked)

def rgb2bin(frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binimg

def read_template(template_path):
    template = cv2.imread(template_path, 0)
    t_w, t_h = template.shape[::-1]
    return template, t_w, t_h

def minimap_template_matching(minimap, template, t_wh, box, big_box, threshold=0.7):
    res = cv2.matchTemplate(
        cv2.cvtColor(minimap, cv2.COLOR_RGB2GRAY),
        template,
        cv2.TM_CCOEFF_NORMED
    )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    inside = False
    above = (max_val >= threshold)

    top_left = max_loc  # (x, y)
    x_c, y_c = (top_left[0] + t_wh[0]//2, top_left[1] + t_wh[1]//2)
    inside = cv2.pointPolygonTest(box, (x_c, y_c), False) >= 0
    inside_big = cv2.pointPolygonTest(big_box, (x_c, y_c), False) >= 0

    return above, inside, inside_big, x_c, y_c
    
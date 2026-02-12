import numpy as np
import cv2

from utils.template import load_template, read_template


class Frame:
    def __init__(self, sct, monitor, args, frame_res=(720, 1280)):
        self.sct = sct
        self.game_res = frame_res

        self.args = args

        self.h_offset, self.w_offset = get_offsets(monitor, self.game_res)
        self.monitor_res = get_monitor_res(monitor)
        self.game_region = {
            "left": self.w_offset,
            "top": self.h_offset,
            "width": self.game_res[1],
            "height": self.game_res[0]
        }
        self.map_length = 312
        self.minimap_region = {
            "left": self.w_offset + self.game_res[1] - self.map_length,
            "top": self.h_offset + self.game_res[0] - self.map_length,
            "width": self.map_length,
            "height": self.map_length
        }
        self.gold_region = {
            "left": self.w_offset + 1100,
            "top": self.h_offset + 1056,
            "width": 57,
            "height": 18
        }
        self.time_region = {
            "left": self.w_offset + 1858,
            "top": self.h_offset + 4,
            "width": 44,
            "height": 19
        }
        self.level_region = {
            "left": self.w_offset + 731,
            "top": self.h_offset + 1051,
            "width": 17,
            "height": 17
        }
        self.health_region = {
            "left": self.w_offset + 776,
            "top": self.h_offset + 1056,
            "width": 273,
            "height": 1
        }
        self.champion_template, self.champion_template_w, self.champion_template_h = read_template(args.template_path)
        self.level_template = load_template(args.level_template_path)
        self.time_template = load_template(args.time_template_path)
        self.gold_template = load_template(args.gold_template_path)

    def _capture(self, region, shape=None, save=""):
        # region["left"] += self.x_offset
        # region["top"] += self.y_offset
        img = np.array(self.sct.grab(region))
        # .COLOR_2RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if shape:
            img = cv2.resize(img, shape)
        if save:
            cv2.imwrite(save, img)
        return img
    
    def process_game_frame(self, game_frame):
        images = dict()
        images["minimap"] = self.minimap_from_frame(game_frame)
        images["gold"] = self.gold_from_frame(game_frame)
        images["level"] = self.level_from_frame(game_frame)
        images["health"] = self.health_from_frame(game_frame)
        images["time"] = self.time_from_frame(game_frame)

        images["game"] = cv2.resize(game_frame, self.args.img_shape)
        return images

    def capture_game_frame(self, shape=None, save=""):
        return self._capture(self.game_region, shape=shape, save=save)

    def capture_minimap(self, shape=None, save=""):
        return self._capture(self.minimap_region, shape=shape, save=save)
    
    def minimap_from_frame(self, frame):
        return frame[
            self.minimap_region["top"]:self.minimap_region["top"] + self.minimap_region["height"],
            self.minimap_region["left"]:self.minimap_region["left"] + self.minimap_region["width"],
            :
        ]

    def capture_gold(self, shape=None, save=""):
        return self._capture(self.gold_region, shape=shape, save=save)
    
    def gold_from_frame(self, frame):
        return frame[
            self.gold_region["top"]:self.gold_region["top"] + self.gold_region["height"],
            self.gold_region["left"]:self.gold_region["left"] + self.gold_region["width"],
            :
        ]
    
    def capture_time(self, shape=None, save=""):
        return self._capture(self.time_region, shape=shape, save=save)
    
    def time_from_frame(self, frame):
        return frame[
            self.time_region["top"]:self.time_region["top"] + self.time_region["height"],
            self.time_region["left"]:self.time_region["left"] + self.time_region["width"],
            :
        ]

    def capture_level(self, shape=None, save=""):
        return self._capture(self.level_region, shape=shape, save=save)
    
    def level_from_frame(self, frame):
        return frame[
            self.level_region["top"]:self.level_region["top"] + self.level_region["height"],
            self.level_region["left"]:self.level_region["left"] + self.level_region["width"],
            :
        ]
    
    def capture_health(self, shape=None, save=""):
        return self._capture(self.health_region, shape=shape, save=save)
    
    def health_from_frame(self, frame):
        return frame[
            self.health_region["top"]:self.health_region["top"] + self.health_region["height"],
            self.health_region["left"]:self.health_region["left"] + self.health_region["width"],
            :
        ]


def get_offsets(monitor, game_res):
    h_offset = monitor["top"] + (monitor["height"] - game_res[0]) // 2
    w_offset = monitor["left"] + (monitor["width"] - game_res[1]) // 2
    return h_offset, w_offset

def get_monitor_res(monitor):
    return (monitor["height"], monitor["width"])

def get_raw_offsets(monitor):
    h_offset = monitor["top"] 
    w_offset = monitor["left"]
    return h_offset, w_offset


if __name__ == "__main__":
    import mss
    sct = mss.mss()
    monitor = sct.monitors[1]
    frame_loader = Frame(sct, monitor)
    print(frame_loader.sct.monitors)
    img = frame_loader._capture({"left": 100, "top": 300, "width": 500, "height": 300})
    cv2.imwrite("test.png", img)

    
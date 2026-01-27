import numpy as np
import cv2


class Frame:
    def __init__(self, sct, monitor, frame_res=(720, 1280)):
        self.sct = sct
        self.game_res = frame_res

        self.h_offset, self.w_offset = get_offsets(monitor, self.game_res)
        self.monitor_res = get_monitor_res(monitor)
        self.game_region = {
            "left": self.w_offset,
            "top": self.h_offset,
            "width": self.game_res[1],
            "height": self.game_res[0]
        }
        self.map_length = 210
        self.map_region = {
            "left": self.w_offset + self.game_res[1] - self.map_length,
            "top": self.h_offset + self.game_res[0] - self.map_length,
            "width": self.map_length,
            "height": self.map_length
        }

    def _capture(self, region, shape=None, save=""):
        # region["left"] += self.x_offset
        # region["top"] += self.y_offset
        img = np.array(self.sct.grab(region))
        # .COLOR_2RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        if shape:
            img = cv2.resize(img, shape)
        if save:
            cv2.imwrite(save, img)
        return img
    
    def capture_game_frame(self, shape=None, save=""):
        return self._capture(self.game_region, shape=shape, save=save)

    def capture_minimap(self, shape=None, save=""):
        return self._capture(self.map_region, shape=shape, save=save)


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

    
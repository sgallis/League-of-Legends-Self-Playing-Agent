import pynput
from pynput.mouse import Button
from pynput.keyboard import Key
import time

from utils.screen import get_offsets, get_monitor_res, get_raw_offsets

class Controller:
    def __init__(self, monitor, window_res=(720, 1280)):
        self.mouse = pynput.mouse.Controller()
        # self.button = pynput.mouse.Button
        self.keyboard = pynput.keyboard.Controller()
        # self.key = pynput.keyboard.Key

        self.h_offset, self.w_offset = get_offsets(monitor, window_res)
        self.game_res = window_res
        self.monitor_res = get_monitor_res(monitor)

        self.raw_h_offset, self.raw_w_offset = get_raw_offsets(monitor)
    
    def compute_mouse_position(self, x, y):
        """Compute desired mouse position inside the window given x and y

        Args:
            x (float): desired x mouse position in the window between 0 and 1
            y (float): desired y mouse position in the window between 0 and 1

        Returns:
            tuple[int, int]
        """
        return (int(self.w_offset + x * self.game_res[1]), int(self.h_offset + y * self.game_res[0]))
    
    def left_click(self, x, y, delay=0, count=1):
        self.mouse.position = self.compute_mouse_position(x, y)
        time.sleep(delay)
        self.mouse.click(Button.left, count=count)
    
    def right_click(self, x, y, delay=0):
        self.mouse.position = self.compute_mouse_position(x, y)
        time.sleep(delay)
        self.mouse.click(Button.right)
    
    def raw_left_click(self, x, y):
        self.mouse.position = (x + self.raw_w_offset, y + self.raw_h_offset)
        self.mouse.click(Button.left)
    
    def type(self, string):
        self.keyboard.type(string)
    
    def press_release(self, key, interval=0):
        self.keyboard.press(key)
        time.sleep(interval)
        self.keyboard.release(key)


if __name__ == "__main__":
    import mss
    sct = mss.mss()
    monitor = sct.monitors[1]
    controller = Controller(monitor, window_res=(576, 1024))
    # press play
    controller.raw_left_click(546, 257)
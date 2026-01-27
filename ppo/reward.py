import cv2
import numpy as np
import torch
import torch.nn.functional as F


BRW = [10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5,
       35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5]
mid_box = np.array([
        [94, 140],
        [69, 113],
        [118, 67],
        [143, 93]
    ], dtype=np.float32)

class RewardModel:
    def __init__(self, args):
        self.args = args
        self.template = cv2.imread("assets/MissFortune_map.png", 0)
        self.t_w, self.t_h = self.template.shape[::-1]
        self.threshold = args.template_threshold
        self.clear()
    
    def clear(self):
        self.gold = 500
        self.alive = True
        self.level = 1
        self.game_time = 0
        self.game_info = {}

    def update_from_game_info(self, game_data):
        self.game_info["gold"] = game_data["activePlayer"]["currentGold"]
        
        game_data_player = game_data["allPlayers"][0]

        self.game_info["alive"] = not game_data_player["isDead"]
        self.game_info["level"] = game_data_player["level"]
        self.game_info["game_time"] = game_data["gameData"]["gameTime"]

    def get_reward(self, game_data, minimap):
        self.update_from_game_info(game_data)

        gold_reward = self.get_gold_reward()
        level_reward = self.get_level_reward()
        dead_reward = self.get_dead_reward()
        pos_reward = self.get_position_reward(minimap)

        reward = gold_reward + level_reward + dead_reward + pos_reward
        return reward
    
    def get_gold_reward(self):
        gold_reward = 0
        curr_gold = self.game_info["gold"]
        gold_diff = curr_gold - self.gold
        if gold_diff > 10:
            gold_reward = gold_diff
        self.gold = curr_gold
        return self.args.r_gold * gold_reward

    def get_level_reward(self):
        level_reward = 0
        game_level = self.game_info["level"]
        if game_level > self.level:
            level_reward = game_level - self.level
            self.level = game_level
        return self.args.r_level * level_reward

    def get_dead_reward(self):
        dead_reward = 0
        game_alive = self.game_info["alive"]
        if self.alive and not game_alive:
            self.alive = False
            dead_reward -= BRW[self.game_info["level"]-1] * 102 / 30
        elif not self.alive and game_alive:
            self.alive = True
        return self.args.r_dead * dead_reward
    
    def get_position_reward(self, img):
        minimap = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(minimap, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        inside = False
        if max_val >= self.threshold:
            top_left = max_loc  # (x, y)
            x_c, y_c = (top_left[0] + self.t_w//2, top_left[1] + self.t_h//2)
            inside = cv2.pointPolygonTest(mid_box, (x_c, y_c), False) >= 0
        pos_r = 0
        time_diff = self.game_info["game_time"] - self.game_time
        pos_r = self.args.r_pos * time_diff * int(inside)
        self.game_time = self.game_info["game_time"]
        return pos_r
    
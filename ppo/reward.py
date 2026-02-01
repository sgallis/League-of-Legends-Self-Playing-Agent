import cv2
import numpy as np

from utils.template import read_template, template_matching
from utils.variables import BRW, mid_box, big_box


class RewardModel:
    def __init__(self, args):
        self.args = args

        self.template, self.t_w, self.t_h = read_template(args.template_path)
        self.threshold = args.template_threshold
        self.clear()
    
    def clear(self):
        self.gold = 500
        self.alive = True
        self.level = 1
        self.game_time = 15
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
        # true gold value per level ~580
        level_reward = 0
        game_level = self.game_info["level"]
        if game_level > self.level:
            level_reward = game_level - self.level
            self.level = game_level
        return self.args.r_level * 50 * level_reward

    def get_dead_reward(self):
        dead_reward = 0
        game_alive = self.game_info["alive"]
        if self.alive and not game_alive:
            self.alive = False
            dead_reward -= BRW[self.game_info["level"]-1] * 102 / 30
        elif not self.alive and game_alive:
            self.alive = True
        return self.args.r_dead * dead_reward
    
    def get_position_reward(self, minimap):
        above, inside, inside_big, x_c, y_c = template_matching(
            minimap,
            self.template,
            (self.t_w, self.t_h),
            mid_box,
            big_box,
            threshold=self.threshold
            )

        pos_r = 0
        time_diff = self.game_info["game_time"] - self.game_time
        # sign = 1 if inside else -1
        sign = 0 if inside else (-1/2 if inside_big else -1)
        pos_r = self.args.r_pos * time_diff * sign
        self.game_time = self.game_info["game_time"]
        return pos_r
    
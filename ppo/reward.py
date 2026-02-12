import cv2
import numpy as np

from utils.template import read_template, minimap_template_matching
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

    def get_reward(self, game_data):
        gold_reward = self.get_gold_reward(game_data)
        level_reward = self.get_level_reward(game_data)
        dead_reward = self.get_dead_reward(game_data)
        pos_reward = self.get_position_reward(game_data)

        reward = gold_reward + level_reward + dead_reward + pos_reward
        return reward
    
    def get_gold_reward(self, game_data):
        gold_reward = 0
        curr_gold = game_data["activePlayer"]["gold"]
        gold_diff = curr_gold - self.gold
        if gold_diff > 10:
            gold_reward = gold_diff
        self.gold = curr_gold
        return self.args.r_gold * gold_reward

    def get_level_reward(self, game_data):
        # true gold value per level ~580
        level_reward = 0
        game_level = game_data["activePlayer"]["level"]
        if game_level > self.level:
            level_reward = game_level - self.level
            self.level = game_level
        return self.args.r_level * 50 * level_reward

    def get_dead_reward(self, game_data):
        dead_reward = 0
        game_alive = game_data["activePlayer"]["health"] > 0
        if self.alive and not game_alive:
            self.alive = False
            dead_reward -= BRW[game_data["activePlayer"]["level"]-1] * 102 / 30
        elif not self.alive and game_alive:
            self.alive = True
        return self.args.r_dead * dead_reward
    
    def get_position_reward(self, game_data):
        pos_r = 0
        time_diff = game_data["game"]["time"] - self.game_time
        # sign = 1 if inside else -1
        sign = -1
        if game_data["activePlayer"]["inside"] or (not game_data["activePlayer"]["health"] > 0):
            sign = 0
        elif game_data["activePlayer"]["insideBig"]:
            sign = -0.5
        pos_r = self.args.r_pos * time_diff * sign
        self.game_time = game_data["game"]["time"]
        return pos_r
    
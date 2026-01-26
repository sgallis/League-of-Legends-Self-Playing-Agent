BRW = [10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5,
       35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5]

class RewardModel:
    def __init__(self, game, args):
        self.args = args
        self.game = game
        
        self.clear()
    
    def clear(self):
        self.gold = 500
        self.alive = True
        self.level = 1
        self.game_info = {}

    def update_from_game_info(self):
        game_data = self.game.get_live_game_data()

        self.game_info["gold"] = game_data["activePlayer"]["currentGold"]
        
        game_data_player = game_data["allPlayers"][0]

        self.game_info["alive"] = not game_data_player["isDead"]
        self.game_info["level"] = game_data_player["level"]
        self.game_info["game_time"] = game_data["gameData"]["gameTime"]

    def get_reward(self):
        self.update_from_game_info()

        gold_reward = self.get_gold_reward()
        level_reward = self.get_level_reward()
        dead_reward = self.get_dead_reward()

        reward = self.args.r_gold * gold_reward + self.args.r_level * level_reward + self.args.r_dead * dead_reward
        return reward
    
    def get_gold_reward(self):
        gold_reward = 0
        curr_gold = self.game_info["gold"]
        gold_diff = curr_gold - self.gold
        if gold_diff > 10:
            gold_reward = gold_diff
        self.gold = curr_gold
        return gold_reward

    def get_level_reward(self):
        level_reward = 0
        game_level = self.game_info["level"]
        if game_level > self.level:
            level_reward = game_level - self.level
            self.level = game_level
        return level_reward

    def get_dead_reward(self):
        dead_reward = 0
        game_alive = self.game_info["alive"]
        if self.alive and not game_alive:
            self.alive = False
            dead_reward -= BRW[self.game_info["level"]-1] * 102 / 30
        elif not self.alive and game_alive:
            self.alive = True
        return dead_reward

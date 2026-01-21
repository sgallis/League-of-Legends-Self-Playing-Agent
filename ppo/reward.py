class RewardModel:
    def __init__(self):
        self.gold = 500
    
    def get_reward(self, game):
        reward = 0

        curr_gold = game.get_player_gold()
        gold_diff = curr_gold - self.gold
        if gold_diff > 10:
            reward += gold_diff / 500

        self.gold = curr_gold
        # print(self.gold, reward)
        return reward

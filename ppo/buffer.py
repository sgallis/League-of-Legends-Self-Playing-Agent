import torch
from torch.utils.data import Dataset


class RolloutBufferDataset(Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        self.clear()
        
    def clear(self):
        self.start = 0
        self.obs = []
        self.game_datas = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.values = []
        self.returns = torch.tensor([])
        self.advantages = torch.tensor([])
        self.size = 0

    def add(self, obs, game_data, action, reward, logp, value):
        self.obs.append(obs)
        self.game_datas.append(game_data)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logps.append(logp)
        self.values.append(value)
        self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "obs": self.obs[idx][0],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "logps": self.logps[idx],
            "advantages": self.advantages[idx],
            "returns": self.returns[idx],
        }

    def compute_rewards(self, reward_model):
        reward_model.clear()

        for t in range(self.start, len(self.rewards)):
            self.rewards[t] += reward_model.get_reward(self.game_datas[t])
            # clear memory
            self.game_datas[t] = None

        return sum(self.rewards[self.start:])

    def compute_advantages_and_returns(self, gamma=0.99, lam=0.95, normalize=True):
        T = len(self.rewards[self.start:])
        advantages = torch.zeros(T)
        returns = torch.zeros(T)
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            # GAE
            delta = self.rewards[self.start+t] + gamma * next_value - self.values[self.start+t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
            next_value = self.values[self.start+t]

        returns = advantages + torch.tensor(self.values[self.start:])

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = torch.cat((self.advantages, advantages))
        self.returns = torch.cat((self.returns, returns))

        old_start = self.start
        self.start = len(self.rewards)

import torch


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.returns = []
        self.advantages = []

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (self.obs[idx],
                self.actions[idx],
                self.logps[idx],
                self.rewards[idx]
                )

    def add(self, obs, action, logp, reward):
        self.obs.append(obs)
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(reward)

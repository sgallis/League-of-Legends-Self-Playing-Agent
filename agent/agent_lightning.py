import time
import logging
import mss
import random

import torch
from torch import optim
from torch.utils.data import DataLoader
import lightning as L

from game.game import Game
from game.client import Client

from agent.agent import Agent
from agent.policy.policy import Policy
from game.interface import Interface

from ppo.loss import ppo_loss
from ppo.buffer import RolloutBufferDataset
from ppo.reward import RewardModel

class AgentLightning(L.LightningModule):
    def __init__(self, args):
        super(AgentLightning, self).__init__()
        self.args = args
        
        sct = mss.mss()
        monitor = sct.monitors[args.monitor_idx]

        self.actions = ["nothing", "move_click"]
        # self.actions = ["move_click"]
        self.actions_specs = {"move_click": 2}

        self.client = Client(monitor, client_res=args.client_res)
        self.game = Game(sct, monitor, game_res=args.game_res)
        self.interface = Interface(self.client, self.game, args)

        self.policy = Policy(self.actions, self.actions_specs)
        self.agent = Agent(monitor, self.game, self.policy, args)
        self.buffer = RolloutBufferDataset()
        self.reward_model = RewardModel(self.game, args)

    def forward(self, x):
        return self.policy(x)

    def on_fit_start(self):
        self.collect_rollouts()

    def on_train_epoch_start(self):
        if self.current_epoch > 0:
            self.collect_rollouts()

    # def on_train_epoch_end(self):
    #     self.buffer.clear()
    #     time.sleep(5)
    #     self.collect_rollout()
    #     metric = sum(self.buffer.rewards)
    #     self.log(
    #         "val/episode_return",
    #         metric,
    #         prog_bar=True,
    #         sync_dist=True,
    #     )

    def collect_rollouts(self):
        self.buffer.clear()
        print(f"Collecting {self.args.games_per_epoch} rollouts!")
        for i in range(self.args.games_per_epoch):
            time.sleep(5)
            self.collect_rollout()
            print(f"Finished collecting rollout {i+1}!")
        print(f"Finished collecting {self.args.games_per_epoch} rollouts!")

    def collect_rollout(self):
        self.reward_model.clear()
        # start env and collect trajectory in buffer
        blue_side = random.random()>0.5
        self.interface.start_custom_game(blue_side)
        self.agent.predefined_start()
        self.agent.collect_trajectory(self.buffer, self.reward_model, self.device)
        self.interface.end_custom_game()

        # compute returns and advantages
        rewards = self.buffer.compute_advantages_and_returns()
        print(f"Rollout rewards {rewards}")

    def train_dataloader(self):
        return DataLoader(
            self.buffer,
            batch_size=self.args.batch_size,
            shuffle=True
        )

    def training_step(self, batch, batch_idx):
        loss, policy_loss, value_loss, entropy, returns = ppo_loss(
            self,
            batch,
            self.actions,
            self.actions_specs
            )
        self.log_dict(
        {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy
        },
        prog_bar=True
        )
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

import time
import logging
import mss
import random

import torch
from torch import optim
from torch.utils.data import DataLoader
import lightning as L

from utils.utils import find_league_monitor

from game.game import Game
from game.client import Client

from agent.agent import Agent
from agent.policy.policy import Policy, DiscretePolicy
from game.interface import Interface

from ppo.loss import ppo_loss, discrete_ppo_loss
from ppo.buffer import RolloutBufferDataset
from ppo.reward import RewardModel

class AgentLightning(L.LightningModule):
    def __init__(self, backbone, args):
        super(AgentLightning, self).__init__()
        self.args = args
        
        sct = mss.mss()
        monitor = find_league_monitor(sct.monitors)

        self.actions = ["nothing", "move_click"]
        # self.actions = ["move_click"]
        self.actions_specs = {"move_click": 2}
        self.m_pos = args.m_pos
        self.n_m_pos = self.m_pos[0] * self.m_pos[1]

        self.client = Client(monitor, client_res=args.client_res)
        self.game = Game(sct, monitor, game_res=args.game_res)
        self.interface = Interface(self.client, self.game, args)

        self.policy = DiscretePolicy(self.actions, self.n_m_pos, backbone)
        self.policy.train()
        self.agent = Agent(monitor, self.game, self.policy, args)
        self.buffer = RolloutBufferDataset()
        self.reward_model = RewardModel(args)

    def forward(self, x):
        return self.policy(x)

    def on_fit_start(self):
        self.collect_rollouts()

    def on_train_epoch_start(self):
        if self.current_epoch > 0:
            self.collect_rollouts()

    def run_validation_rollout(self, train=True, sample=True):
        self.buffer.clear()
        time.sleep(5)
        rewards = self.collect_rollout(train=train, sample=sample)
        logging.info(f"Validation rollout rewards: {rewards}")
        return rewards

    def collect_rollouts(self):
        r_list = []

        self.buffer.clear()
        logging.info(f"Collecting {self.args.games_per_epoch} rollouts!")
        for i in range(self.args.games_per_epoch):
            time.sleep(3)
            rewards = self.collect_rollout()
            r_list.append(rewards)
            logging.info(f"Rollout {i+1} rewards: {rewards}")
        self.ep_return = sum(r_list) / len(r_list)
        logging.info(f"Finished collecting {self.args.games_per_epoch} rollouts!")

    def collect_rollout(self, train=True, sample=True):
        # start env and collect trajectory in buffer
        # blue_side = random.random()>0.5
        blue_side = False
        game_start_time = self.interface.start_custom_game(blue_side)
        self.agent.predefined_start(game_start_time)
        self.agent.collect_trajectory(
            game_start_time,
            self.buffer,
            self.device,
            train=train,
            sample=sample
            )
        self.interface.end_custom_game()
        if train:
            # compute rewards
            rewards = self.buffer.compute_rewards(self.reward_model)
            # compute returns and advantages
            self.buffer.compute_advantages_and_returns()
            return rewards
        return None

    def train_dataloader(self):
        return DataLoader(
            self.buffer,
            batch_size=self.args.batch_size,
            shuffle=True
        )

    def on_train_batch_start(self, batch, batch_idx):
        assert self.policy.net.training is False, "Expected backbone to be in eval()"
        assert self.policy.net.fc.training is True, "Expected head (fc) to be in train()"

    def training_step(self, batch, batch_idx):
        # loss, policy_loss, value_loss, entropy, returns = ppo_loss(
        #     self,
        #     batch,
        #     self.actions,
        #     self.actions_specs
        #     )
        loss, policy_loss, value_loss, entropy, returns = discrete_ppo_loss(
            self,
            batch,
            )
        self.log_dict(
        {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "episode_return": self.ep_return
        },
        prog_bar=True
        )
        if torch.cuda.is_available():
            self.log("gpu/mem_alloc_mb",
                     torch.cuda.memory_allocated() / 1024**2,
                     prog_bar=True)
        return loss
    
    def test_dataloader(self):
        return DataLoader([0])  # dummy single batch

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

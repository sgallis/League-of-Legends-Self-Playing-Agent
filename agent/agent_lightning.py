import time
import logging
import mss

import lightning as L

from agent.agent import Agent
from agent.policy.policy import Policy
from game.interface import Interface

from ppo.loss import ppo_loss

class AgentLightning(L.LightningModule):
    def __init__(self, args):
        self.args = args
        
        sct = mss.mss()
        monitor = sct.monitors[args.monitor_idx]

        self.actions = ["nothing", "move_click"]
        self.actions_specs = {"move_click": 2}

        self.interface = Interface(sct, monitor, args)
        self.policy = Policy(self.actions, self.actions_specs)
        self.agent = Agent(monitor, self.policy, args)

    def training_step(self, batch, batch_idx):
        loss, _, _, _, returns = ppo_loss(
            self,
            batch,
            self.actions,
            self.action_specs
            )
        logging.info()
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical, Normal
from torchvision.models import resnet18, ResNet18_Weights

from utils.utils import squash
from agent.policy.net.policy_net import PolicyNet
from ppo.loss import ppo_loss


class Policy(nn.Module):
    def __init__(
            self,
            actions,
            action_specs,
            backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            ):
        super(Policy, self).__init__()
        self.actions = actions
        self.action_specs = action_specs
        self.net = backbone
        # freeze backbone
        for p in self.net.parameters():
            p.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = PolicyNet(
            self.actions,
            self.action_specs,
            in_features,
            hidden_dim=in_features//2
            )

    def forward(self, x):
        return self.net(x)

    def sample_action(self, img):
        # batch size always = 1
        value, action_logits, action_params = self(img)

        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        logp_action = action_dist.log_prob(action)

        logp_param = 0

        action_coords = None
        # continuous parameters
        if self.actions[action.item()] in self.action_specs.keys():
            mean, std = action_params[self.actions[action.item()]]
            param_dist = Normal(mean, std)
            raw = param_dist.sample()
            action_coords = squash(raw)
            logp_param = (
                param_dist.log_prob(raw)
                - torch.log(1 - torch.tanh(raw)**2 + 1e-6)
            ).sum(-1)
        
        # joint log-prob
        logp = (logp_action + logp_param).item()

        # action as tuple (action, param1, param2, ...)
        action = torch.tensor((action.item(), 0.0, 0.0))
        if action_coords is not None:    
            action_coords = action_coords.detach().cpu()
            action = torch.tensor((action[0], *action_coords[0]))
        # 0.01, tensor([1.0000, 0.3244, 0.7226], dtype=torch.float64), -0.04081709682941437
        return value.item(), action, logp
    
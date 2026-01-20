import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.policy.net.backbone import Backbone

class Net(nn.Module):
    def __init__(self, actions, action_specs, hidden_dim=256):
        """
        action_specs: dict
            key: action index
            value: number of continuous parameters
        Example:
            {
                0: 0,  # do nothing
                1: 2,  # move (x, y)
                2: 1,  # rotate (theta)
            }
        """
        super(Net, self).__init__()
        self.num_actions = len(actions)
        self.action_specs = action_specs

        self.backbone = Backbone(32)

        self.value_fc = nn.Linear(512, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        self.action_fc = nn.Linear(512, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, self.num_actions)

        # Parameter heads per action
        self.param_means = nn.ModuleDict()
        self.param_logstds = nn.ParameterDict()

        for a, dim in action_specs.items():
            if dim > 0:
                self.param_means[a] = nn.Linear(hidden_dim, dim)
                self.param_logstds[a] = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        z = self.backbone(x)

        value = self.value_head(F.relu(self.value_fc(z)))

        z_a = F.relu(self.action_fc(z))
        action_logits = self.action_head(z_a)

        action_params = {}
        for a, dim in self.action_specs.items():
            if dim > 0:
                mean = self.param_means[a](z_a)
                std = self.param_logstds[a].exp()
                action_params[a] = (mean, std)

        return value.squeeze(-1), action_logits, action_params
    
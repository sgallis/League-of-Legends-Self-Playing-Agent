import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from utils.utils import unsquash


def ppo_loss(
    model,
    batch,
    possible_actions,
    actions_specs,
    clip_eps=0.2,
    vf_coef=1,
    ent_coef=0.001,
):
    obs = batch["obs"]
    # (action_id, coord1, coord2, ...)
    actions = batch["actions"]
    cat_actions = actions[:, 0]
    old_logps = batch["logps"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    values, actions_logits, actions_params = model(obs)

    actions_dist = Categorical(logits=actions_logits)
    new_logps = actions_dist.log_prob(cat_actions)
    entropy = actions_dist.entropy()

    for i, a in enumerate(actions):
        a_name = possible_actions[int(a[0].item())]
        if a_name in actions_specs.keys():
            mean = actions_params[a_name][0][i]
            std = actions_params[a_name][1]

            param_dist = Normal(mean, std)

            entropy[i] += param_dist.entropy().sum(-1)

            action_values = unsquash(a[1:])
            logp_param = (
                param_dist.log_prob(action_values)
                - torch.log(1 - torch.tanh(action_values)**2 + 1e-6)
            ).sum(-1)
            new_logps[i] += logp_param
    ratio = torch.exp(new_logps - old_logps)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    actor_loss = -torch.min(ratio*advantages, clipped*advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = entropy.mean()
    
    loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

    return loss, actor_loss, value_loss, entropy, returns

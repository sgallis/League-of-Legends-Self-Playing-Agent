from contextlib import contextmanager

import torch


@contextmanager
def inference_mode(module):
    was_training = module.training
    module.eval()
    with torch.no_grad():
        yield
    module.train(was_training)


def squash(raw):
    return 0.5 * (torch.tanh(raw) + 1.0)

def unsquash(x, eps=1e-6):
    x = torch.clamp(x, eps, 1 - eps)
    return torch.atanh(2 * x - 1)

def mouse_id_to_grid_id(m_id, m_pos):
    divide = m_pos[1]
    grid_x = m_id % divide 
    grid_y = m_id // divide
    return grid_x, grid_y

def find_league_monitor(monitors):
    for m in monitors:
        if m["left"] == 0 and m["top"] == 0:
            return m
    raise NotImplementedError

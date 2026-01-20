import torch


def squash(raw):
    return 0.5 * (torch.tanh(raw) + 1.0)

def unsquash(x, eps=1e-6):
    x = torch.clamp(x, eps, 1 - eps)
    return torch.atanh(2 * x - 1)

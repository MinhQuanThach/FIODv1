import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_optimisers(model, lr=1e-4, weight_decay=5e-4, momentum=0.9):
    return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

def get_lr_schedulers(optimiser, total_steps):
    return CosineAnnealingLR(optimiser, T_max=total_steps)
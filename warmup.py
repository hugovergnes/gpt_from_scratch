import math
import torch

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.

    During the warmup phase, the learning rate increases linearly from 0 to the base learning rate.
    After the warmup phase, it decays following a cosine schedule down to a minimum learning rate.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for the linear warmup.
        total_steps (int): Total number of training steps (including warmup).
        min_lr (float, optional): The minimum learning rate after decay. Defaults to 0.0.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.

    Example:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=500, total_steps=10000, min_lr=1e-5)
        for batch_idx, (x, y) in enumerate(train_loader):
            ...
            optimizer.step()
            scheduler.step()
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        current_step = self.last_epoch + 1
        if current_step < self.warmup_steps:
            return [base_lr * current_step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            progress = (current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(torch.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
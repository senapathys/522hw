from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(
            self, optimizer, last_epoch=-1, max_epochs=100, start_lr=0.1, end_lr=0.0001
    ):
        """
        Create a new scheduler.

        Arguments:
        optimizer (torch.optim.Optimizer): Optimizer to use for learning rate updates.
        last_epoch (int): The index of the last epoch. Default: -1.
        max_epochs (int): Maximum number of epochs for training. Default: 100.
        start_lr (float): Starting learning rate. Default: 0.1.
        end_lr (float): Ending learning rate. Default: 0.0001.
        """
        self.max_epochs = max_epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.total_steps = self.max_epochs * len(optimizer.param_groups)
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Compute the current learning rate
        progress = self.last_epoch / self.max_epochs
        current_lr = self.end_lr + (self.start_lr - self.end_lr) * (1 - progress) ** 2
        return [current_lr for _ in self.optimizer.param_groups]

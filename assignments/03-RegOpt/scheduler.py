from typing import List
import math
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, cycle_len=1, max_lr=0.1, min_lr=None):
        """
        Create a new scheduler.

        :param cycle_len: The number of epochs in a cycle. The learning rate will go from min_lr to max_lr and back
        to min_lr over the course of cycle_len epochs.
        :param max_lr: The maximum learning rate.
        :param min_lr: The minimum learning rate. Defaults to max_lr / 10.
        """
        self.cycle_len = cycle_len
        self.max_lr = max_lr
        self.min_lr = min_lr if min_lr is not None else max_lr / 10

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to student: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        cycle = math.floor(1 + self.last_epoch / (2 * self.cycle_len))
        x = abs(self.last_epoch / self.cycle_len - 2 * cycle + 1)
        lr = self.min_lr + (self.max_lr - self.min_lr) * max(0, 1 - x)

        return [lr for _ in self.base_lrs]

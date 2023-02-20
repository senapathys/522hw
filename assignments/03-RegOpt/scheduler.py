from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom LR scheduler
    """

    def __init__(
        self,
        optimizer,
        base_lr=0.001,
        max_lr=0.003,
        step_size=2000,
        mode="triangular",
        gamma=1.0,
        last_epoch=-1,
    ):
        """
        Create a new scheduler.

        Arguments:
        - optimizer: optimizer that will be updated by this scheduler
        - base_lr: initial learning rate
        - max_lr: maximum learning rate
        - step_size: number of iterations for half a cycle
        - mode: "triangular" or "triangular2" (default is "triangular")
        - gamma: multiplier for decreasing the maximum learning rate at each cycle (default is 1.0)
        - last_epoch: index of the last epoch (default is -1)
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        gets learning rate
        """
        cycle = self.last_epoch // (2 * self.step_size)
        x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
        if self.mode == "triangular":
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)
        elif self.mode == "triangular2":
            lr = (
                self.base_lr
                + (self.max_lr - self.base_lr) * max(0, 1 - x) * self.gamma**cycle
            )
        else:
            raise ValueError("unexpected mode: " + self.mode)
        return [lr for _ in self.optimizer.param_groups]

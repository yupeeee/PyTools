import math
from torch.optim.lr_scheduler import _LRScheduler


__all__ = [
    "warmup_wrapper",
]

strategies = [
    "const",
    "cos",
    "linear",
]


def warmup_wrapper(scheduler, config, start_epoch):
    """
    in config.yaml:
    ⋮
    SCHEDULER:
      TYPE: ...
      ⋮
      WARMUP:
        INIT_LR: float
        WARMUP_STEPS: int
        STRATEGY: str
    ⋮
    """
    return WarmUpLR(
        scheduler=scheduler,
        init_lr=config.SCHEDULER.WARMUP.INIT_LR,
        warmup_steps=config.SCHEDULER.WARMUP.WARMUP_STEPS,
        strategy=config.SCHEDULER.WARMUP.STRATEGY,
        start_epoch=start_epoch,
    )


def warmup_const(start, end, pct):
    return start if pct < 1 else end


def warmup_cos(start, end, pct):
    cos_pct = (math.cos(math.pi * (1 + pct)) + 1) / 2

    return start + (end - start) * cos_pct


def warmup_linear(start, end, pct):
    return start + (end - start) * pct


class WarmUpLR(_LRScheduler):
    def __init__(
            self,
            scheduler: _LRScheduler,
            init_lr: float = 0.,
            warmup_steps: int = 1,
            strategy: str = "linear",
            start_epoch: int = 1,
    ) -> None:
        assert strategy in strategies

        self.scheduler = scheduler
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.strategy = strategy

        self.start_epoch = start_epoch

        if strategy == "const":
            self.warmup_ft = warmup_const
        elif strategy == "cos":
            self.warmup_ft = warmup_cos
        elif strategy == "linear":
            self.warmup_ft = warmup_linear
        else:
            raise ValueError

        self.init_param()

    def __getattr__(self, name):
        return getattr(self.scheduler, name)

    def state_dict(self):
        wrapper_state_dict = {
            key: value for key, value in self.__dict__.items()
            if (key != "optimizer" and key != "scheduler")
        }
        wrapped_state_dict = {
            key: value for key, value in self.scheduler.__dict__.items()
            if key != "optimizer"
        }

        return {"wrapped": wrapped_state_dict, "wrapper": wrapper_state_dict}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict["wrapper"])
        self.scheduler.__dict__.update(state_dict["wrapped"])

    def init_param(self):
        for group in self.scheduler.optimizer.param_groups:
            group["warmup_max_lr"] = group["lr"]

            init_lr = min(self.init_lr, group["lr"])

            group["lr"] = init_lr
            group["warmup_init_lr"] = init_lr

        for i in range(self.start_epoch - 1):
            self.scheduler.optimizer.step()
            self.step()

    def get_lr(self):
        lrs = []
        step_num = self._step_count

        if step_num <= self.warmup_steps:
            for group in self.scheduler.optimizer.param_groups:
                lr = self.warmup_ft(
                    start=group["warmup_init_lr"],
                    end=group["warmup_max_lr"],
                    pct=step_num / self.warmup_steps,
                )
                lrs.append(lr)

        else:
            lrs = self.scheduler.get_lr()

        return lrs

    def step(self, *args):
        if self._step_count <= self.warmup_steps:
            lrs = self.get_lr()

            for param_group, lr in zip(
                self.scheduler.optimizer.param_groups, lrs
            ):
                param_group["lr"] = lr

            self._step_count += 1

        self.scheduler.step(*args)

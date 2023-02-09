from torch.optim import lr_scheduler


__all__ = [
    "build_scheduler",
]

schedulers = [
    # "LambdaLR",
    # "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    # "ConstantLR",
    # "LinearLR",
    # "ExponentialLR",
    # "PolynomialLR",
    "CosineAnnealingLR",
    # "SequentialLR",
    # "CyclicLR",
    # "OneCycleLR",

    # "ChainedScheduler",
    # "ReduceLROnPlateau",
    # "CosineAnnealingWarmRestarts",
]


def build_scheduler(optimizer, config):
    scheduler_type = config.SCHEDULER.TYPE

    assert scheduler_type in schedulers

    if scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, config)

    elif scheduler_type == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, config)

    elif scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, config)

    else:
        raise ValueError

    if hasattr(config.SCHEDULER, "WARMUP"):
        from .lr_warmup import warmup_wrapper

        scheduler = warmup_wrapper(scheduler, config)

    return scheduler


def StepLR(optimizer, config):
    """
    in config.yaml:
    ⋮
    SCHEDULER:
      TYPE: StepLR
      STEP_SIZE: int
      GAMMA: float
    ⋮
    """
    return lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config.SCHEDULER.STEP_SIZE,
        gamma=config.SCHEDULER.GAMMA,
        last_epoch=-1,
        verbose=False,
    )


def MultiStepLR(optimizer, config):
    """
    in config.yaml:
    ⋮
    SCHEDULER:
      TYPE: MultiStepLR
      MILESTONES: Iterable[int]
      GAMMA: float
    ⋮
    """
    return lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.SCHEDULER.MILESTONES,
        gamma=config.SCHEDULER.GAMMA,
        last_epoch=-1,
        verbose=False,
    )


def CosineAnnealingLR(optimizer, config):
    """
    in config.yaml:
    ⋮
    SCHEDULER:
      TYPE: CosineAnnealingLR
      T_MAX: int
      ETA_MIN: float
    ⋮
    """
    return lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.SCHEDULER.T_MAX,
        eta_min=config.SCHEDULER.ETA_MIN,
        last_epoch=-1,
        verbose=False,
    )

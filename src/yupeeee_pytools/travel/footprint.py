from typing import Tuple

import torch

from .config import *


class Footprint:
    def __init__(
            self,
            model,
            step: int,
            method: str = default_method,
            normalize: str = default_normalize,
            seed: int = default_seed,
            use_cuda: bool = False,
            verbose: bool = False,
    ) -> None:
        from .direction import DirectionGenerator

        self.model = model
        self.step = step
        self.method = method
        self.normalize = normalize
        self.seed = seed
        self.use_cuda = use_cuda
        self.verbose = verbose

        self.direction_generator = DirectionGenerator(
            method=method,
            normalize=normalize,
            seed=seed,
            model=model,
            use_cuda=use_cuda,
        )

    def generate_footprints(
            self,
            data: torch.Tensor,
            directions: torch.Tensor,
            epsilons: torch.Tensor,
    ):
        from ..tools import repeat_tensor

        _data = repeat_tensor(
            tensor=data,
            repeat=self.step + 1,
            dim=0,
        )

        _directions = repeat_tensor(
            tensor=directions,
            repeat=self.step + 1,
            dim=0,
        )

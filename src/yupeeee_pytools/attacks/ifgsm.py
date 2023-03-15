import torch
import tqdm


__all__ = [
    "IFGSM",
]


class IFGSM:
    def __init__(
            self,
            model,
            epsilon: float,
            iteration: int = 1,
            aid: bool = False,
            normalizer: torch.nn.Module = None,
            clip_per_iter: bool = True,
            use_cuda: bool = False,
    ) -> None:
        assert epsilon >= 0.
        assert iteration > 0

        self.model = model
        self.epsilon = epsilon
        self.iteration = iteration
        self.aid = aid
        self.normalizer = normalizer
        self.clip_per_iter = clip_per_iter
        self.use_cuda = use_cuda

        self.alpha = epsilon / iteration
        self.machine = "cuda" if use_cuda else "cpu"

    def __call__(
            self,
            data: torch.Tensor,
            targets: torch.Tensor,
            verbose: bool = False,
    ) -> torch.Tensor:
        if self.epsilon == 0.:
            return data

        _data = data.detach()

        for _ in tqdm.trange(
                self.iteration,
                desc=f"I-FGSM {'aid' if self.aid else 'attack'} (ϵ={self.epsilon})",
                disable=not verbose,
        ):
            _data = self.fgsm(_data, targets, self.alpha)

            if self.clip_per_iter:
                _data = _data.clamp(0, 1)

        return _data

    def get_data_grad(
            self,
            data: torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        data = getattr(data, self.machine)()
        targets = getattr(targets, self.machine)()

        if self.normalizer is not None:
            data = self.normalizer(data)

        data = data.detach()
        data.requires_grad = True

        out = self.model(data)
        self.model.zero_grad()

        loss = torch.nn.CrossEntropyLoss()(out, targets)
        loss.backward()

        with torch.no_grad():
            data_grad = data.grad.data

        return data_grad.detach().cpu()

    def fgsm(
            self,
            data: torch.Tensor,
            targets: torch.Tensor,
            epsilon: float,
    ) -> torch.Tensor:
        if epsilon == 0.:
            return data

        sign_data_grad = self.get_data_grad(data, targets).sign()

        if self.aid:
            sign_data_grad = -sign_data_grad

        perturbations = epsilon * sign_data_grad
        perturbations = perturbations.clamp(-epsilon, epsilon)

        _data = data + perturbations

        return _data

    def generate_universal_perturbation(
            self,
            data: torch.Tensor,
            targets: torch.Tensor,
            batch_size: int = None,
            verbose: bool = False,
    ) -> torch.Tensor:
        from ..tools.misctools import repeat_tensor

        num_data = len(data)

        universal_perturbation = torch.zeros(size=data.shape[1:])

        if self.epsilon == 0.:
            return universal_perturbation

        if batch_size is None:
            data_batch = data.unsqueeze(dim=0)
            targets_batch = targets.unsqueeze(dim=0)
        else:
            data_batch = torch.split(data, batch_size, dim=0)
            targets_batch = torch.split(targets, batch_size, dim=0)

        num_batch = len(data_batch)

        for _ in tqdm.trange(
                self.iteration,
                desc=f"Universal I-FGSM {'aid' if self.aid else 'attack'} (ϵ={self.epsilon})",
                disable=not verbose,
        ):
            sum_data_grad = torch.zeros_like(universal_perturbation)

            for i in range(num_batch):
                data_mini_batch = data_batch[i]
                targets_mini_batch = targets_batch[i]

                universal_perturbations = repeat_tensor(
                    universal_perturbation,
                    repeat=len(data_mini_batch),
                    dim=0,
                )

                _data_mini_batch = data_mini_batch + universal_perturbations

                if self.clip_per_iter:
                    _data_mini_batch = _data_mini_batch.clamp(0, 1)

                data_grad = self.get_data_grad(_data_mini_batch, targets_mini_batch)
                sum_data_grad += torch.sum(data_grad, dim=0)

            sign_data_grad = (sum_data_grad / num_data).sign()

            if self.aid:
                sign_data_grad = -sign_data_grad

            universal_perturbation += self.alpha * sign_data_grad
            universal_perturbation = universal_perturbation.clamp(-self.epsilon, self.epsilon)

        return universal_perturbation.detach().cpu()

methods = [
    "fgsm",
    "random",
]

normalizations = [
    "dim",
    "unit",
]

default_method = "fgsm"
default_normalize = "dim"
default_seed = None

eps_for_incorrect = -1.
eps_for_divergence = -1.

invalid_epsilons = [
    eps_for_incorrect,
    eps_for_divergence,
]

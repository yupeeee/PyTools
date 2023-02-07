activation_replacements = [
    ("GELU", "ReLU"),
    ("ReLU", "GELU"),
]

normalization_replacements = [
    ("BatchNorm2d", "LayerNorm"),
    ("LayerNorm", "BatchNorm2d"),
]

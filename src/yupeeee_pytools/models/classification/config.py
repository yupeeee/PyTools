import torchvision.transforms as tf

from ...datasets import normalize
from ...tools import merge_lists_in_dictionary


modes = [None, "train", "eval"]

weight_specifications = [None, "DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"]
default_weight_specification = "IMAGENET1K_V1"

default_cifar10_train_preprocess = tf.Compose([
    # tf.ToPILImage(),
    # tf.RandomCrop(32, padding=4),
    # tf.RandomHorizontalFlip(),
    # tf.RandomRotation(15),
    # tf.Resize(224),
    tf.ToTensor(),
    normalize["CIFAR-10"],
])

default_cifar10_val_preprocess = tf.Compose([
    # tf.ToPILImage(),
    # tf.Resize(224),
    tf.ToTensor(),
    normalize["CIFAR-10"],
])

default_cifar100_train_preprocess = tf.Compose([
    # tf.ToPILImage(),
    # tf.RandomCrop(32, padding=4),
    # tf.RandomHorizontalFlip(),
    # tf.RandomRotation(15),
    # tf.Resize(224),
    tf.ToTensor(),
    normalize["CIFAR-100"],
])

default_cifar100_val_preprocess = tf.Compose([
    # tf.ToPILImage(),
    # tf.Resize(224),
    tf.ToTensor(),
    normalize["CIFAR-100"],
])

default_imagenet_preprocess = tf.Compose([
    tf.CenterCrop(256),
    tf.Resize(224),
    tf.ToTensor(),
    normalize["ImageNet"],
])

pytorch_imagenet_models = {
    "Alexnet": [
        "alexnet",
    ],

    "ConvNeXt": [
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    ],

    "DenseNet": [
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
    ],

    "EfficientNet": [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
    ],

    "EfficientNetV2": [
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
    ],

    "GoogLeNet": [
        "googlenet",
    ],

    "Inception v3": [
        "inception_v3",
    ],

    "MaxVit": [
        # "maxvit_t",
    ],

    "MNASNet": [
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_0",
        "mnasnet1_3",
    ],

    "MobileNet V2": [
        "mobilenet_v2",
    ],

    "MobileNet V3": [
        "mobilenet_v3_large",
        "mobilenet_v3_small",
    ],

    "RegNet": [
        "regnet_y_400mf",
        "regnet_y_800mf",
        "regnet_y_1_6gf",
        "regnet_y_3_2gf",
        "regnet_y_8gf",
        "regnet_y_16gf",
        "regnet_y_32gf",
        # "regnet_y_128gf",
        "regnet_x_1_6gf",
        "regnet_x_3_2gf",
        "regnet_x_8gf",
        "regnet_x_16gf",
        "regnet_x_32gf",
    ],

    "ResNet": [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ],

    "ResNext": [
        "resnext50_32x4d",
        "resnext101_32x8d",
        "resnext101_64x4d"
    ],

    "ShuffleNet V2": [
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
    ],

    "SqueezeNet": [
        "squeezenet1_0",
        "squeezenet1_1",
    ],

    "SwinTransformer": [
        "swin_t",
        "swin_s",
        "swin_b",
        # "swin_v2_t",
        # "swin_v2_s",
        # "swin_v2_b",
    ],

    "VGG": [
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
    ],

    "VisionTransformer": [
        "vit_b_16",
        "vit_b_32",
        "vit_l_16",
        "vit_l_32",
        # "vit_h_14",
    ],

    "Wide ResNet": [
        "wide_resnet50_2",
        "wide_resnet101_2",
    ],
}

pytorch_imagenet_model_names = {
    "alexnet": "AlexNet",

    "convnext_tiny": "ConvNeXt_Tiny",
    "convnext_small": "ConvNeXt_Small",
    "convnext_base": "ConvNeXt_Base",
    "convnext_large": "ConvNeXt_Large",

    "densenet121": "DenseNet121",
    "densenet161": "DenseNet161",
    "densenet169": "DenseNet169",
    "densenet201": "DenseNet201",

    "efficientnet_b0": "EfficientNet_B0",
    "efficientnet_b1": "EfficientNet_B1",
    "efficientnet_b2": "EfficientNet_B2",
    "efficientnet_b3": "EfficientNet_B3",
    "efficientnet_b4": "EfficientNet_B4",
    "efficientnet_b5": "EfficientNet_B5",
    "efficientnet_b6": "EfficientNet_B6",
    "efficientnet_b7": "EfficientNet_B7",

    "efficientnet_v2_s": "EfficientNet_V2_S",
    "efficientnet_v2_m": "EfficientNet_V2_M",
    "efficientnet_v2_l": "EfficientNet_V2_L",

    "googlenet": "GoogLeNet",

    "inception_v3": "Inception_V3",

    # "maxvit_t",

    "mnasnet0_5": "MNASNet0_5",
    "mnasnet0_75": "MNASNet0_75",
    "mnasnet1_0": "MNASNet1_0",
    "mnasnet1_3": "MNASNet1_3",

    "mobilenet_v2": "MobileNet_V2",

    "mobilenet_v3_large": "MobileNet_V3_Large",
    "mobilenet_v3_small": "MobileNet_V3_Small",

    "regnet_y_400mf": "RegNet_Y_400MF",
    "regnet_y_800mf": "RegNet_Y_800MF",
    "regnet_y_1_6gf": "RegNet_Y_1_6GF",
    "regnet_y_3_2gf": "RegNet_Y_3_2GF",
    "regnet_y_8gf": "RegNet_Y_8GF",
    "regnet_y_16gf": "RegNet_Y_16GF",
    "regnet_y_32gf": "RegNet_Y_32GF",
    "regnet_y_128gf": "RegNet_Y_128GF",
    "regnet_x_1_6gf": "RegNet_X_1_6GF",
    "regnet_x_3_2gf": "RegNet_X_3_2GF",
    "regnet_x_8gf": "RegNet_X_8GF",
    "regnet_x_16gf": "RegNet_X_16GF",
    "regnet_x_32gf": "RegNet_X_32GF",

    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "resnet152": "ResNet152",

    "resnext50_32x4d": "ResNeXt50_32X4D",
    "resnext101_32x8d": "ResNeXt101_32X8D",
    "resnext101_64x4d": "ResNeXt101_64X4D",

    "shufflenet_v2_x0_5": "ShuffleNet_V2_X0_5",
    "shufflenet_v2_x1_0": "ShuffleNet_V2_X1_0",
    "shufflenet_v2_x1_5": "ShuffleNet_V2_X1_5",
    "shufflenet_v2_x2_0": "ShuffleNet_V2_X2_0",

    "squeezenet1_0": "SqueezeNet1_0",
    "squeezenet1_1": "SqueezeNet1_1",

    "swin_t": "Swin_T",
    "swin_s": "Swin_S",
    "swin_b": "Swin_B",
    # "swin_v2_t",
    # "swin_v2_s",
    # "swin_v2_b",

    "vgg11": "VGG11",
    "vgg11_bn": "VGG11_BN",
    "vgg13": "VGG13",
    "vgg13_bn": "VGG13_BN",
    "vgg16": "VGG16",
    "vgg16_bn": "VGG16_BN",
    "vgg19": "VGG19",
    "vgg19_bn": "VGG19_BN",

    "vit_b_16": "ViT_B_16",
    "vit_b_32": "ViT_B_32",
    "vit_l_16": "ViT_L_16",
    "vit_l_32": "ViT_L_32",
    # "vit_h_14": "ViT_H_14",

    "wide_resnet50_2": "Wide_ResNet50_2",
    "wide_resnet101_2": "Wide_ResNet101_2",
}

not_in_pytorch_imagenet_models = {
    "DeiT": [
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
        "deit_tiny_distilled_patch16_224",
        "deit_small_distilled_patch16_224",
        "deit_base_distilled_patch16_224",
        "deit_base_patch16_384",
        "deit_base_distilled_patch16_384",
    ],

    "PoolFormer": [
        "poolformer_s12",
        "poolformer_s24",
        "poolformer_s36",
        "poolformer_m36",
        "poolformer_m48",
    ],

    "PVT": [
        "pvt_tiny",
        "pvt_small",
        "pvt_medium",
        "pvt_large",
    ],

    "MLP-Mixer": [
        "mixer_b_16",
        "mixer_l_16",
    ],
}

not_in_pytorch_imagenet_model_names = {
    "deit_tiny_patch16_224": "DeiT-Ti",
    "deit_small_patch16_224": "DeiT-S",
    "deit_base_patch16_224": "DeiT-B",
    "deit_tiny_distilled_patch16_224": "DeiT-Ti⚗",
    "deit_small_distilled_patch16_224": "DeiT-S⚗",
    "deit_base_distilled_patch16_224": "DeiT-B⚗",
    "deit_base_patch16_384": "DeiT-B↑384",
    "deit_base_distilled_patch16_384": "DeiT-B⚗↑384",

    "poolformer_s12": "PoolFormer-S12",
    "poolformer_s24": "PoolFormer-S24",
    "poolformer_s36": "PoolFormer-S36",
    "poolformer_m36": "PoolFormer-M36",
    "poolformer_m48": "PoolFormer-M48",

    "pvt_tiny": "PVT-Tiny",
    "pvt_small": "PVT-Small",
    "pvt_medium": "PVT-Medium",
    "pvt_large": "PVT-Large",

    "mixer_b_16": "Mixer-B_16",
    "mixer_l_16": "Mixer-L_16",
}

list_of_pytorch_imagenet_models = merge_lists_in_dictionary(
    pytorch_imagenet_models
)

list_of_not_in_pytorch_imagenet_models = merge_lists_in_dictionary(
    not_in_pytorch_imagenet_models
)

imagenet_models = \
    list_of_pytorch_imagenet_models + list_of_not_in_pytorch_imagenet_models

imagenet_model_names = {**pytorch_imagenet_model_names, **not_in_pytorch_imagenet_model_names}

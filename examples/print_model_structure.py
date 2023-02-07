from argparse import ArgumentParser
import torch

from pytools.models import ImageNetClassificationModel, default_weight_specification, weights_dir
from pytools.tools import save_list_in_txt


save_path = "./pytools/models/classification/structure"


def run():
    model = ImageNetClassificationModel(
        name=model_name,
        pretrained=False,
        specify_weights=default_weight_specification,
        weights_dir=weights_dir,
        mode="eval",
        use_cuda=use_cuda,
    )

    dummy_data = getattr(torch.zeros(size=(1, 3, 224, 224)), machine)()
    layers = model.dissect(dummy_data)

    del dummy_data

    layer_names = list(layers.keys())

    structure = ""

    for layer_name in layer_names:
        structure += f"{layer_name}\n"
        structure += f"{layers[layer_name]}\n\n"

    save_list_in_txt(
        [structure],
        path=save_path,
        save_name=model_name,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-model", type=str)
    parser.add_argument("-cuda", action="store_true", default=False)

    args = parser.parse_args()

    model_name = args.model
    use_cuda = args.cuda
    machine = "cuda" if use_cuda else "cpu"

    run()

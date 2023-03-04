from typing import Any, Dict

from argparse import ArgumentParser
import torch
import tqdm

from yupeeee_pytools.datasets import ImageNetDataset, normalize, denormalize
from yupeeee_pytools.models import ImageNetClassificationModel, Preprocess, default_weight_specification
from yupeeee_pytools.tools import makedir, angle_of_three_points, save_dictionary_in_csv
from yupeeee_pytools.travel import fgsm_direction


weights_dir = None      # directory containing weights of not-in-pytorch models


def run():
    save_path = f"./travel_data/ImageNet_fgsm_travel_eps_{eps}_seed_{seed}/{model_name}"
    makedir(f"{save_path}/magnitude")
    makedir(f"{save_path}/direction")
    makedir(f"{save_path}/cumulative_magnitude")
    makedir(f"{save_path}/cumulative_direction")

    dataset = ImageNetDataset(
        root="D:/dataset/ImageNet",
        split="val",
        transform=Preprocess(
            model_name=model_name,
            specify_weights=default_weight_specification,
        ),
        target_transform=None,
    )
    data = dataset.data

    classes = list(set([int(v) for v in dataset.targets]))
    classes.sort()

    normalizer = normalize[dataset.name]
    denormalizer = denormalize[dataset.name]

    model = ImageNetClassificationModel(
        name=model_name,
        pretrained=True,
        specify_weights=default_weight_specification,
        weights_dir=weights_dir,
        mode="eval",
        use_cuda=use_cuda,
    )

    dummy_data = getattr(torch.zeros_like(dataset[0][0].unsqueeze(dim=0)), machine)()
    layers = model.dissect(dummy_data)

    del dummy_data

    layer_names = list(layers.keys())

    for c in classes:
        indices = torch.arange(c * 50, c * 50 + 50)
        image_names = [data[i][0].split('\\')[-1].split('/')[-1] for i in indices]
        class_nos = [c] * len(image_names)

        # initialize travel data
        mag_data: Dict[str, Any] = {"img": image_names, "class": class_nos}
        dir_data: Dict[str, Any] = {"img": image_names, "class": class_nos}
        cu_mag_data: Dict[str, Any] = {"img": image_names, "class": class_nos}      # cumulative
        cu_dir_data: Dict[str, Any] = {"img": image_names, "class": class_nos}      # cumulative

        for layer_name in layer_names:
            mag_data[layer_name] = []
            dir_data[layer_name] = []
            cu_mag_data[layer_name] = []
            cu_dir_data[layer_name] = []

        # travel
        for i in tqdm.tqdm(
                indices,
                desc=f"FGSM-travel of {model_name} on {dataset.name} (Class {c})...",
        ):
            image, target = dataset[i]

            image = getattr(image.unsqueeze(dim=0), machine)()
            target = torch.Tensor([target]).to(torch.int64)

            direction = getattr(fgsm_direction(
                data=image,
                targets=target,
                model=model.model,
                normalize="dim",
                seed=seed,
                use_cuda=use_cuda,
            ), machine)()

            _image = (image + direction * eps).detach()
            __image = (_image + direction * eps).detach()
            direction_norm = torch.norm(direction.reshape(-1) * eps).detach().cpu()

            inputs, outputs = model.x_ray(image)
            _inputs, _outputs = model.x_ray(_image)
            __inputs, __outputs = model.x_ray(__image)

            for layer_name in layer_names:
                layer = layers[layer_name]

                x = getattr(inputs[layer_name], machine)()
                _x = getattr(_inputs[layer_name], machine)()
                __x = _x + (_x - x)

                y = layer(x)
                _y = layer(_x)
                __y = layer(__x)

                mag_standard = torch.norm((_x - x).reshape(-1)).detach().cpu()
                magnitude_change = float(torch.norm((_y - y).detach().cpu()) / mag_standard)

                direction_change = float(torch.pi - angle_of_three_points(
                    start=y.reshape(-1),
                    mid=_y.reshape(-1),
                    end=__y.reshape(-1),
                    eps=1e-6,
                ).detach().cpu())

                mag_data[layer_name].append(magnitude_change)
                dir_data[layer_name].append(direction_change)

                # cumulative
                out = outputs[layer_name].reshape(-1)
                _out = _outputs[layer_name].reshape(-1)
                __out = __outputs[layer_name].reshape(-1)

                cumulative_magnitude_change = float(torch.norm(_out - out) / direction_norm)

                cumulative_direction_change = float(torch.pi - angle_of_three_points(
                    start=out,
                    mid=_out,
                    end=__out,
                    eps=1e-6,
                ))

                cu_mag_data[layer_name].append(cumulative_magnitude_change)
                cu_dir_data[layer_name].append(cumulative_direction_change)

        save_dictionary_in_csv(
            dictionary=mag_data,
            save_dir=f"{save_path}/magnitude",
            save_name=f"{c}",
            index_col="img",
        )

        save_dictionary_in_csv(
            dictionary=dir_data,
            save_dir=f"{save_path}/direction",
            save_name=f"{c}",
            index_col="img",
        )

        save_dictionary_in_csv(
            dictionary=cu_mag_data,
            save_dir=f"{save_path}/cumulative_magnitude",
            save_name=f"{c}",
            index_col="img",
        )

        save_dictionary_in_csv(
            dictionary=cu_dir_data,
            save_dir=f"{save_path}/cumulative_direction",
            save_name=f"{c}",
            index_col="img",
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-model", type=str)
    parser.add_argument("-eps", type=float, default=0.01)
    parser.add_argument("-seed", type=int, default=0)
    parser.add_argument("-cuda", action="store_true", default=False)

    args = parser.parse_args()

    model_name = args.model
    eps = args.eps
    seed = args.seed
    use_cuda = args.cuda
    machine = "cuda" if use_cuda else "cpu"

    run()

import torchvision.transforms as tf

from pytools import datasets


root = "D:/dataset"


for dataset_name in datasets.datasets:
    if dataset_name == "ImageNet":
        print(f"Extracting {dataset_name} dataset in {root}/{dataset_name}...")

        dataset = datasets.ImageNetDataset(
            root=f"{root}/{dataset_name}",
            split='val',
            transform=tf.Compose([
                tf.Resize(256),
                tf.CenterCrop(224),
                tf.ToTensor(),
                # datasets.normalize[dataset_name],
            ]),
            target_transform=int,
        )

        print(f"Extraction successful.")

        print("\n- - - - - - - - - - - - - - - - - - - -\n"
              "Dataset Info"
              "\n- - - - - - - - - - - - - - - - - - - -\n")

        print(f"[Dataset] {dataset_name}")
        print(f"[Directory] {root}/{dataset_name}")

        print(f"[NUM_DATA] {len(dataset)}")
        print(f"[NUM_CLASS] {len(set([int(v) for v in dataset.targets]))}")

        data, target = dataset[0]
        print(f"[DATA_SHAPE] {data.shape}")
        print(f"[DATA_TYPE] {type(data)}")
        print(f"[TARGET_TYPE] {type(target)}")

        print("\n- - - - - - - - - - - - - - - - - - - -\n")

    else:
        print(f"Downloading {dataset_name} dataset to {root}/{dataset_name}...")

        dataset = getattr(datasets, f"{''.join(dataset_name.split('-'))}Dataset")(
            root=f"{root}/{dataset_name}",
            train=True,
            transform=tf.Compose([
                tf.ToTensor(),
                # datasets.normalize[dataset_name],
            ]) if dataset_name in ["CIFAR-10", "CIFAR-100"] else None,
            target_transform=int,
            download=True,
        )

        print(f"Download successful.")

        print("\n- - - - - - - - - - - - - - - - - - - -\n"
              "Dataset Info"
              "\n- - - - - - - - - - - - - - - - - - - -\n")

        print(f"[Dataset] {dataset_name}")
        print(f"[Directory] {root}/{dataset_name}")

        print(f"[NUM_DATA] {len(dataset)}")
        print(f"[NUM_CLASS] {len(set([int(v) for v in dataset.targets]))}")

        data, target = dataset[0]
        print(f"[DATA_SHAPE] {data.shape}")
        print(f"[DATA_TYPE] {type(data)}")
        print(f"[TARGET_TYPE] {type(target)}")

        mean, std = dataset.mean_and_std_of_data()
        print(f"[MEAN_DATA] {mean}")
        print(f"[STD_DATA] {std}")

        print("\n- - - - - - - - - - - - - - - - - - - -\n")

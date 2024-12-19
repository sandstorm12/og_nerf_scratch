import os
import yaml
import torch
import argparse
import numpy as np


from metadata_loader import load_metadata, load_images, get_transforms
from image_2_ray import image_2_ray_sample


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, images, transforms, num_samples):
        self._images = images
        self._transforms = transforms
        self._num_samples = num_samples

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        transform = self._transforms[idx]

        focal_length = .5 * image.shape[1] / np.tan(.5 * 0.6911112070083618)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        directions, colors = image_2_ray_sample(
            image, focal_length, rotation, translation, self._num_samples)
        
        ray_directions = torch.from_numpy(directions)
        ray_source = torch.from_numpy(
            np.tile(translation, (self._num_samples, 1)))
        ray_colors = torch.from_numpy(colors).to(torch.float32)

        return ray_directions, ray_source, ray_colors


# Just for test
if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print("Configs:", configs)

    metadata_train, metadata_val, metadata_test = load_metadata(configs)
    images_train = load_images(metadata_train, os.path.dirname(configs['train']))
    transforms_train = get_transforms(metadata_train)
    num_samples = 100

    dataset = NeRFDataset(images_train, transforms_train, 100)
    print("Dataset len:", dataset.__len__())
    item = dataset.__getitem__(0)
    print("Dataset sample:", item[0].shape, item[1].shape, item[2].shape)

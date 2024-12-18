import os
import cv2
import yaml
import json
import argparse
import numpy as np


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


def load_metadata(configs):
    # Load the train transform
    with open(configs['train'], 'r') as handle:
        metadata_train = json.load(handle)

    # Load the val transform
    with open(configs['val'], 'r') as handle:
        metadata_val = json.load(handle)

    # Load the test transform
    with open(configs['test'], 'r') as handle:
        metadata_test = json.load(handle)

    return metadata_train, metadata_val, metadata_test


def load_images(metadata, basedir):
    frames = metadata['frames']
    paths = [frame['file_path'] for frame in frames]
    paths = [os.path.join(basedir, path + '.png') for path in paths]

    images = [cv2.imread(path) for path in paths]

    images = np.asarray(images)

    return images


def get_transforms(metadata):
    frames = metadata['frames']
    transforms = [frame['transform_matrix'] for frame in frames]

    transforms = np.asarray(transforms)

    return transforms


# Just for test
if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print("Configs:", configs)

    metadata_train, metadata_val, metadata_test = load_metadata(configs)

    images_train = load_images(metadata_train, os.path.dirname(configs['train']))
    images_val = load_images(metadata_val, os.path.dirname(configs['val']))
    images_test = load_images(metadata_test, os.path.dirname(configs['test']))

    print(f"Number of training images: {len(images_train)}")
    print(f"Number of validation images: {len(images_val)}")
    print(f"Number of testing images: {len(images_test)}")

    if configs['visualize_images']:
        for image in images_train:
            cv2.imshow("Image", image)
            if cv2.waitKey(0) == ord('q'):
                break

        for image in images_val:
            cv2.imshow("Image", image)
            if cv2.waitKey(0) == ord('q'):
                break

        for image in images_test:
            cv2.imshow("Image", image)
            if cv2.waitKey(0) == ord('q'):
                break

    transforms_train = get_transforms(metadata_train)
    transforms_val = get_transforms(metadata_val)
    transforms_test = get_transforms(metadata_test)

    print(f"Number of training transforms: {len(transforms_train)}")
    print(f"Number of validation transforms: {len(transforms_val)}")
    print(f"Number of testing transforms: {len(transforms_test)}")


    # Some analytics
    translations = np.concatenate(
        (transforms_train[:, :3, 3],
         transforms_val[:, :3, 3],
         transforms_test[:, :3, 3]), axis=0)
    print("translations.shape", translations.shape)

    print(f"Max translation: {np.max(translations)}")
    print(f"Min translation: {np.min(translations)}")


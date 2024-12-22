import os
import cv2
import json
import numpy as np

from config_utils import get_arguments, load_configs

from render import _sample_points_batch
from image_2_ray import image_2_ray_sample


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


def _test_vis_images(metadata_train, metadata_val, metadata_test):
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


def _test_anal_trans(metadata_train, metadata_val, metadata_test):
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

    print(f"Max translation: {np.max(translations, axis=0)}")
    print(f"Min translation: {np.min(translations, axis=0)}")


def _test_vis_translations(metadata_train, metadata_val, metadata_test):
    import open3d as o3d

    transforms_train = get_transforms(metadata_train)
    transforms_val = get_transforms(metadata_val)
    transforms_test = get_transforms(metadata_test)

    rotations = np.concatenate(
        (transforms_train[:, :3, :3],
         transforms_val[:, :3, :3],
         transforms_test[:, :3, :3]), axis=0)
    print("rotations.shape", rotations.shape)

    translations = np.concatenate(
        (transforms_train[:, :3, 3],
         transforms_val[:, :3, 3],
         transforms_test[:, :3, 3]), axis=0)
    print("translations.shape", translations.shape)

    print(translations.shape, translations.dtype)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(translations)

    num_samples = 100
    ray_length = 8
    img_dim = 800
    focal_length = .5 * img_dim / np.tan(.5 * 0.6911112070083618)
    rotation = rotations[2]
    translation = translations[2]

    print(translation, rotation)

    us = np.asarray([0, 0, 0, 400, 400, 400, 800, 800, 800])
    vs = np.asarray([0, 400, 800, 0, 400, 800, 0, 400, 800])
    points, _ = _sample_points_batch(us, vs, num_samples, ray_length,
                            img_dim, focal_length, rotation, translation)
    
    # Distance between first and last point
    dist = np.linalg.norm(points[0, -1] - points[0, 0])
    print("Distance between first and last point:", dist)
    
    pcd_view = o3d.geometry.PointCloud()
    pcd_view.points = o3d.utility.Vector3dVector(np.asarray(points).reshape(-1, 3))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd, pcd_view])


def _test_vis_sample(metadata_train, metadata_val, metadata_test):
    import open3d as o3d

    basedir = os.path.dirname(configs['train'])

    transforms_train = get_transforms(metadata_train)

    rotations = transforms_train[:, :3, :3].astype(np.float64)
    translations = transforms_train[:, :3, 3].astype(np.float64)

    print(translations.shape, translations.dtype)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(translations)

    idx = np.random.randint(0, len(metadata_train['frames']))
    num_samples = 8
    ray_length = 8
    img_dim = 800
    focal_length = .5 * img_dim / np.tan(.5 * 0.6911112070083618)

    image = cv2.imread(os.path.join(basedir, metadata_train['frames'][idx]['file_path'] + '.png'))
    directions, colors = image_2_ray_sample(
        image, focal_length, rotations[idx], translations[idx], num_samples)
    
    steps = np.linspace(0, ray_length, 64)
    points = np.asarray([translations[idx] + directions * step for step in steps])
    points = np.swapaxes(points, 0, 1)
    
    # Distance between first and last point
    dist = np.linalg.norm(points[0, -1] - points[0, 0])
    print("Distance between first and last point:", dist)
    
    pcd_view = o3d.geometry.PointCloud()
    pcd_view.points = o3d.utility.Vector3dVector(np.asarray(points).reshape(-1, 3))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd, pcd_view])


# Just for test
if __name__ == "__main__":
    args = get_arguments()
    configs = load_configs(args.config)

    print("Configs:", configs)

    metadata_train, metadata_val, metadata_test = load_metadata(configs)

    # _test_vis_images(metadata_train, metadata_val, metadata_test)

    # _test_anal_trans(metadata_train, metadata_val, metadata_test)    

    _test_vis_translations(metadata_train, metadata_val, metadata_test)

    # _test_vis_sample(metadata_train, metadata_val, metadata_test)


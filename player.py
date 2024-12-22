import cv2
import math
import torch
import numpy as np

from tqdm import tqdm

from nerf import NeRF
from config_utils import get_arguments, load_configs
from render import render_pixel_batch, sample_points_batch


def rotate_matrix(rotation_matrix, axis, degree):
    """
    Rotates a given rotation matrix along a specified axis by a certain degree.

    Parameters:
        rotation_matrix (numpy.ndarray): A 3x3 rotation matrix.
        axis (str): The axis to rotate along ('x', 'y', or 'z').
        degree (float): The angle in degrees to rotate.

    Returns:
        numpy.ndarray: The rotated rotation matrix.
    """
    # Convert degrees to radians
    radians = np.radians(degree)

    # Define the axis rotation matrices
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(radians), -np.sin(radians)],
            [0, np.sin(radians), np.cos(radians)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(radians), 0, np.sin(radians)],
            [0, 1, 0],
            [-np.sin(radians), 0, np.cos(radians)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(radians), -np.sin(radians), 0],
            [np.sin(radians), np.cos(radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    # Rotate the input matrix
    rotated_matrix = np.dot(R, rotation_matrix)
    
    return rotated_matrix


def _render(img_dim, focal_length, rotation, translation, batch_size, device):
    us = np.asarray([u for _ in range(img_dim) for u in range(img_dim)])
    vs = np.asarray([v for v in range(img_dim) for _ in range(img_dim)])
    with torch.no_grad():
        image = np.zeros((img_dim, img_dim, 3), np.uint8)

        num_batches = math.ceil(len(us) / batch_size)
        for idx in tqdm(range(num_batches)):
            us_batch = us[idx*batch_size:min(len(us), (idx+1)*batch_size)]
            vs_batch = vs[idx*batch_size:min(len(vs), (idx+1)*batch_size)]

            points, directions = sample_points_batch(
                us_batch, vs_batch, configs["num_samples"],
                configs["ray_length"], img_dim, focal_length,
                rotation, translation)
            
            points = torch.tensor(
                points, dtype=torch.float32, device=device)
            directions = torch.tensor(
                directions, dtype=torch.float32, device=device)
            color = render_pixel_batch(
                nerf, points, directions,
                step_size=configs["ray_length"]/configs["num_samples"])
            color = (color.cpu().detach().numpy() * 255)

            image[us_batch, vs_batch] = color

    return image


def _play(device, configs):
    img_dim = configs["img_dim"]
    focal_length = .5 * img_dim / np.tan(.5 * configs["fov"])
    batch_size = configs["batch_size"]

    img_dim //= configs["scaling_factor"]
    focal_length //= configs["scaling_factor"]

    rotation = np.asarray(configs["rotation_init"], dtype=np.float32)
    translation = np.asarray(configs["translation_init"], dtype=np.float32)

    while True:
        image = _render(img_dim, focal_length, rotation,
                        translation, batch_size, device)

        image = cv2.resize(image, configs["view_dims"])
        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord("e"):
            rotation = rotate_matrix(rotation, "x", 1)
        elif key == ord("r"):
            rotation = rotate_matrix(rotation, "x", -1)
        elif key == ord("t"):
            rotation = rotate_matrix(rotation, "y", 1)
        elif key == ord("y"):
            rotation = rotate_matrix(rotation, "y", -1)
        elif key == ord("u"):
            rotation = rotate_matrix(rotation, "z", 1)
        elif key == ord("i"):
            rotation = rotate_matrix(rotation, "z", -1)
        elif key == ord("w"):
            translation[2] += 0.1
        elif key == ord("s"):
            translation[2] -= 0.1
        elif key == ord("a"):
            translation[0] -= 0.1
        elif key == ord("d"):
            translation[0] += 0.1
        elif key == ord("z"):
            translation[1] -= 0.1
        elif key == ord("x"):
            translation[1] += 0.1
        elif key == ord("c"):
            if img_dim == configs["img_dim"]:
                img_dim //= configs["scaling_factor"]
                focal_length //= configs["scaling_factor"]
            else:
                img_dim *= configs["scaling_factor"]
                focal_length *= configs["scaling_factor"]


if __name__ == "__main__":
    args = get_arguments()
    configs = load_configs(args.config)

    print("Configs:", configs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nerf = NeRF()
    nerf.load_state_dict(torch.load("./artifacts/nerf_38_0.0056.pt"))
    nerf.to(device)

    _play(device, configs)

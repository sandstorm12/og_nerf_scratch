import os
import yaml
import torch
import argparse
import numpy as np

from metadata_loader import load_metadata, load_images, get_transforms
from dataloader import NeRFDataset
from nerf import NeRF

from tqdm import tqdm


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


def _render_pixel_batch(nerf, points_batch, directions, step_size):
    rgbas = nerf(points_batch[..., 0], points_batch[..., 1],
                 points_batch[..., 2],
                 directions[..., 0], directions[..., 1])

    rgb, sigma = rgbas[..., :3], rgbas[..., 3]
    
    transmittance = torch.cat([torch.ones_like(sigma[:, :1]), torch.cumprod(
        torch.exp(-torch.clamp(sigma * step_size, max=255.0)), dim=1
    )[:, :-1]], dim=1)

    alpha = 1.0 - torch.exp(-torch.clamp(sigma * step_size, max=255.0))

    weight = transmittance * alpha

    color = torch.sum(weight[..., None] * rgb, dim=1)

    return color


def _sample_points_batch(directions, translation, num_samples, ray_length):
    steps = torch.linspace(0, ray_length, num_samples, dtype=torch.float32)
    points = torch.stack(
        [translation + directions * step for step in steps]).to(torch.float32)
    points = points.permute(1, 0, 2)

    return points


def train(configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    nerf = NeRF()
    nerf.load_state_dict(torch.load("./artifacts/nerf_39_0.0066.pt"))
    nerf.to(device)
    
    metadata_train, _, _ = load_metadata(configs)
    images_train = load_images(metadata_train,
                               os.path.dirname(configs['train']))
    transforms_train = get_transforms(metadata_train)
    num_samples = 100

    optimizer = torch.optim.Adam(nerf.parameters(), lr=5e-4)
    loss_fn = torch.nn.MSELoss()

    dataset = NeRFDataset(images_train, transforms_train, 2048)
    bar = tqdm(range(configs["epochs"]))
    for epoch in bar:
        loss_epoch = 0
        for batch in dataset:
            ray_directions, ray_source, ray_colors = batch
            ray_directions = ray_directions.to(device)
            ray_source = ray_source.to(device)
            ray_colors = ray_colors.to(device)

            points = _sample_points_batch(
                ray_directions, ray_source, num_samples, ray_length=8
            ).to(device)

            color = _render_pixel_batch(nerf, points, ray_directions,
                                        step_size=8/num_samples)

            loss = loss_fn(color, ray_colors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

        bar.set_description(
            f"Epoch: {epoch}, Loss: {loss_epoch / len(dataset):.4f}")

        torch.save(
            nerf.state_dict(),
            f"./artifacts/nerf_{epoch}_{loss_epoch / len(dataset):.4f}.pt")


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print("Configs:", configs)

    train(configs)

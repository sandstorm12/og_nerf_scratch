import math
import torch
import numpy as np

from nerf import NeRF


def render_pixel(nerf, points, direction, step_size):    
    rgbas = nerf(points[:, 0], points[:, 1], points[:, 2], direction[0], direction[1])
    rgb, sigma = rgbas[:, :3], rgbas[:, 3]
    
    transmittance = torch.cumprod(
        torch.exp(-torch.clamp(sigma * step_size, max=10.0)),
        dim=0)
    alpha = 1.0 - torch.exp(-torch.clamp(sigma * step_size, max=10.0))

    weight = transmittance * alpha

    color = torch.sum(weight[..., None] * rgb, dim=0)

    return color


def render_pixel_batch(nerf, points_batch, directions, step_size):
    rgbas = nerf(points_batch[..., 0], points_batch[..., 1],
                 points_batch[..., 2],
                 directions[..., 0], directions[..., 1])

    rgb, sigma = rgbas[..., :3], rgbas[..., 3]
    
    transmittance = torch.cat([torch.ones_like(sigma[:, :1]), torch.cumprod(
        torch.exp(-torch.clamp(sigma * step_size, max=255.0)), dim=1
    )[:, :-1]], dim=1)

    # print(transmittance)

    alpha = 1.0 - torch.exp(-torch.clamp(sigma * step_size, max=255.0))

    weight = transmittance * alpha

    color = torch.sum(weight[..., None] * rgb, dim=1)

    return color


def sample_points(u, v, num_samples, ray_length, img_dim, focal_length,
                   rotation, translation):
    cx = img_dim // 2
    cy = img_dim // 2

    x = (u - cx) / focal_length
    y = -(v - cy) / focal_length
    z = -np.ones_like(x)

    direction = np.asarray([x, y, z])
    direction = np.dot(rotation, direction)
    # direction = np.add(direction, translation)
    direction = direction / np.linalg.norm(direction)

    steps = np.linspace(0, ray_length, num_samples)
    points = np.asarray([translation + direction * step for step in steps])

    return points, direction


def sample_points_batch(u, v, num_samples, ray_length, img_dim, focal_length,
                         rotation, translation):
    cx = img_dim // 2
    cy = img_dim // 2

    x = (u - cx) / focal_length
    y = -(v - cy) / focal_length
    z = -np.ones_like(x)

    directions = np.stack((x, y, z), axis=-1)
    directions = np.dot(directions, rotation.T)
    # directions = np.add(directions, translation)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    steps = np.linspace(0, ray_length, num_samples)
    points = np.asarray([translation + directions * step for step in steps])
    points = np.swapaxes(points, 0, 1)

    return points, directions


def _test_points():
    import open3d as o3d

    num_samples = 100
    ray_length = 8
    img_dim = 800
    focal_length = .5 * img_dim / np.tan(.5 * 0.6911112070083618)
    rotation = np.identity(3)
    translation = np.zeros(3)

    us = [0, 400, 800]
    vs = [0, 400, 800]

    points_all = []
    for u in us:
        for v in vs:
            points, _ = sample_points(u, v, num_samples, ray_length,
                                    img_dim, focal_length, rotation, translation)
            
            points_all += points
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_all))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def _test_points_batch():
    import open3d as o3d

    num_samples = 100
    ray_length = 8
    img_dim = 800
    focal_length = .5 * img_dim / np.tan(.5 * 0.6911112070083618)
    rotation = np.identity(3)
    translation = np.zeros(3)

    us = np.asarray([0, 0, 0, 400, 400,400, 800, 800, 800])
    vs = np.asarray([0, 400, 800, 0, 400, 800, 0, 400, 800])
    points, _ = sample_points_batch(us, vs, num_samples, ray_length,
                            img_dim, focal_length, rotation, translation)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points).reshape(-1, 3))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def _test_render(device):
    import cv2
    from tqdm import tqdm

    num_samples = 100
    ray_length = 8
    img_dim = 800
    focal_length = .5 * img_dim / np.tan(.5 * 0.6911112070083618)
    rotation = np.identity(3)
    translation = np.zeros(3)

    with torch.no_grad():
        image = np.zeros((img_dim, img_dim, 3))
        for u in tqdm(range(img_dim)):
            for v in range(img_dim):
                points, direction = sample_points(u, v, num_samples, ray_length,
                                        img_dim, focal_length, rotation, translation)
                
                points = torch.tensor(points, dtype=torch.float32, device=device)
                direction = torch.tensor(direction, dtype=torch.float32, device=device)
                color = render_pixel(nerf, points, direction, step_size=ray_length/num_samples)
                color = (color.cpu().detach().numpy() * 255)

                image[u, v] = color

    cv2.imshow("image", image)
    cv2.waitKey(0)


def _test_render_batch(device):
    import cv2
    from tqdm import tqdm

    num_samples = 100
    ray_length = 8
    img_dim = 800
    focal_length = .5 * img_dim / np.tan(.5 * 0.6911112070083618)
    rotation = np.asarray(
        [[0.44296363,  0.31377721, -0.83983749],
         [-0.89653969,  0.15503149, -0.41494811],
         [0.,          0.93675458,  0.3499869],])
    translation = np.asarray([-3.38549352, -1.67270947,  1.41084266])

    us = np.asarray([u for _ in range(img_dim) for u in range(img_dim)])
    vs = np.asarray([v for v in range(img_dim) for _ in range(img_dim)])
    with torch.no_grad():
        image = np.zeros((img_dim, img_dim, 3), np.uint8)

        batch_size = 2048
        num_batches = math.ceil(len(us) / batch_size)
        for idx in tqdm(range(num_batches)):
            us_batch = us[idx*batch_size:min(len(us), (idx+1)*batch_size)]
            vs_batch = vs[idx*batch_size:min(len(vs), (idx+1)*batch_size)]

            points, directions = sample_points_batch(us_batch, vs_batch, num_samples, ray_length,
                                    img_dim, focal_length, rotation, translation)
            
            points = torch.tensor(points, dtype=torch.float32, device=device)
            directions = torch.tensor(directions, dtype=torch.float32, device=device)
            color = render_pixel_batch(nerf, points, directions, step_size=ray_length/num_samples)
            color = (color.cpu().detach().numpy() * 255)

            image[us_batch, vs_batch] = color

    print(image.min(), image.mean(), image.max())

    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nerf = NeRF()
    nerf.load_state_dict(torch.load("./artifacts/nerf_38_0.0056.pt"))
    nerf.to(device)

    # _test_points()
    # _test_points_batch()

    # num_samples = 100
    # ray_length = 8
    # img_dim = 800
    # focal_length = .5 * img_dim / np.tan(.5 * 0.6911112070083618)
    # rotation = np.identity(3)
    # translation = np.zeros(3)

    # u = 400
    # v = 400
    # points, direction = sample_points(u, v, num_samples, ray_length,
    #                         img_dim, focal_length, rotation, translation)
    
    # points = torch.tensor(points, dtype=torch.float32)
    # direction = torch.tensor(direction, dtype=torch.float32)
    # render_pixel(nerf, points, direction, step_size=ray_length/num_samples)


    # _test_render(device)
    _test_render_batch(device)

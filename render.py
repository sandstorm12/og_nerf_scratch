import torch
import numpy as np


from nerf import NeRF



def _render_pixel(nerf, points, direction, step_size):    
    rgbas = nerf(points[:, 0], points[:, 1], points[:, 2], direction[0], direction[1])
    rgb, sigma = rgbas[:, :3], rgbas[:, 3]
    
    transmittance = torch.cumprod(
        torch.exp(-torch.clamp(sigma * step_size, max=10.0)),
        dim=0)
    alpha = 1.0 - torch.exp(-torch.clamp(sigma * step_size, max=10.0))

    weight = transmittance * alpha

    color = torch.sum(weight[..., None] * rgb, dim=0)

    return color



def _sample_points(u, v, num_samples, ray_length, img_dim, focal_length, rotation, translation):
    cx = img_dim // 2
    cy = img_dim // 2

    x = (u - cx) / focal_length
    y = (v - cy) / focal_length
    z = np.ones_like(x)

    direction = np.asarray([x, y, z])
    direction = direction / np.linalg.norm(direction)
    direction = np.dot(rotation, direction)
    direction = np.add(direction, translation)

    steps = np.linspace(0, ray_length, num_samples)
    points = np.asarray([translation + direction * step for step in steps])

    return points, direction


def _sample_points_batch(u, v, num_samples, ray_length, img_dim, focal_length, rotation, translation):
    cx = img_dim // 2
    cy = img_dim // 2

    x = (u - cx) / focal_length
    y = (v - cy) / focal_length
    z = np.ones_like(x)

    directions = np.asarray([x, y, z])
    directions = directions / np.linalg.norm(directions)
    directions = np.dot(rotation, directions)
    directions = np.add(directions, translation)

    steps = np.linspace(0, ray_length, num_samples)
    points = np.asarray([translation + directions * step for step in steps])

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
            points, _ = _sample_points(u, v, num_samples, ray_length,
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

    us = [0, 400, 800]
    vs = [0, 400, 800]
    
    points_all = []
    for u in us:
        for v in vs:
            points, _ = _sample_points_batch(u, v, num_samples, ray_length,
                                    img_dim, focal_length, rotation, translation)
            
            points_all += points
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points_all))

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
                points, direction = _sample_points(u, v, num_samples, ray_length,
                                        img_dim, focal_length, rotation, translation)
                
                points = torch.tensor(points, dtype=torch.float32, device=device)
                direction = torch.tensor(direction, dtype=torch.float32, device=device)
                color = _render_pixel(nerf, points, direction, step_size=ray_length/num_samples)
                color = (color.cpu().detach().numpy() * 255)

                image[u, v] = color

    cv2.imshow("image", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nerf = NeRF()
    nerf.to(device)

    # _test_points()

    # num_samples = 100
    # ray_length = 8
    # img_dim = 800
    # focal_length = .5 * img_dim / np.tan(.5 * 0.6911112070083618)
    # rotation = np.identity(3)
    # translation = np.zeros(3)

    # u = 400
    # v = 400
    # points, direction = _sample_points(u, v, num_samples, ray_length,
    #                         img_dim, focal_length, rotation, translation)
    
    # points = torch.tensor(points, dtype=torch.float32)
    # direction = torch.tensor(direction, dtype=torch.float32)
    # _render_pixel(nerf, points, direction, step_size=ray_length/num_samples)


    _test_render(device)

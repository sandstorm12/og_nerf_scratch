import numpy as np


def _image_2_ray_uv(image, u, v, focal_length, rotation, translation):
    cx = image.shape[1] // 2
    cy = image.shape[0] // 2

    x = (u - cx) / focal_length
    y = -(v - cy) / focal_length
    z = -np.ones_like(x)

    direction = np.asarray([x, y, z])
    direction = np.dot(rotation, direction)
    # direction = -1 * np.add(direction, translation)

    colors = image[v, u]

    return direction, colors


def image_2_ray_sample(image, focal_length, rotation, translation, num_samples):
    u = np.random.randint(0, image.shape[1], num_samples)
    v = np.random.randint(0, image.shape[0], num_samples)

    cx = image.shape[1] // 2
    cy = image.shape[0] // 2

    x = (u - cx) / focal_length
    y = -(v - cy) / focal_length
    z = -np.ones_like(x)

    directions = np.stack((x, y, z), axis=-1)
    directions = np.dot(directions, rotation.T)
    # directions = np.add(directions, translation)
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    colors = image[v, u] / 255.

    return directions, colors


# Just for test
if __name__ == '__main__':
    import cv2

    image = cv2.imread("/home/hamid/Documents/indie_projects/og_nerf_scratch/data/nerf_synthetic/lego/train/r_0.png")

    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

    rotation = np.identity(3)
    translation = np.zeros(3)
    focal_length = .5 * image.shape[1] / np.tan(.5 * 0.6911112070083618)
    
    u = image.shape[1] // 2
    v = image.shape[0] // 2
    direction, color = _image_2_ray_uv(image, u, v, focal_length, rotation, translation)
    print("direction", direction)
    print("color", color)


    directions, colors = image_2_ray_sample(image, focal_length, rotation, translation, 100)
    print("directions", directions.shape)
    print("colors", colors.shape)

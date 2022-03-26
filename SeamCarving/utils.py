import os
from typing import Dict, Any

import numpy as np
from PIL import Image

NDArray = Any
Options = Any


def open_image(image_path: str):
    """

    :param image_path: the path of the input image
    :return: NDArray that represent the image. The dtype is set to float32.
    """
    pil_image = Image.open(image_path)
    numpy_image = np.array(pil_image, dtype=np.float32)
    assert numpy_image.ndim == 3, 'We only support RGB images in this assignment'
    return numpy_image


def to_grayscale(image: NDArray):
    """Converts an RGB image to grayscale image."""
    assert image.ndim == 3 and image.shape[2] == 3
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])


def normalize_image(image: NDArray):
    """Normalize image pixels to be between [0., 1.0]"""
    min_img = image.min()
    max_img = image.max()
    normalized_image = (image - min_img) / (max_img - min_img)
    normalized_image *= 255.0
    return normalized_image


def get_gradients(image: NDArray):
    """
    Returns the image gradients.
    :param image: The input RGB image.
    :return: A grayscale [0., 255.0] image which represents the image gradients.
    """
    # Convert image to grayscale first!
    if image.ndim == 3:
        image = to_grayscale(image)
    shift_y = np.roll(image, -1, axis=0)
    shift_y[-1, ...] = image[-2, ...]
    shift_x = np.roll(image, -1, axis=1)
    shift_x[:, -1, ...] = image[:, -2, ...]
    grads = np.sqrt(0.5 * (shift_x - image) ** 2 + 0.5 * (shift_y - image) ** 2)
    return grads


def save_images(images: Dict[str, NDArray], outdir: str, prefix: str = 'img'):
    """A helper method that saves a dictionary of images"""

    def _prepare_to_save(image: NDArray):
        """Helper method that converts the image to Uint8"""
        if image.dtype == np.uint8:
            return image
        return normalize_image(image).astype(np.uint8)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for image_name, image_data in images.items():
        Image.fromarray(_prepare_to_save(image_data)).save(f'{outdir}/{prefix}_{image_name}.png')

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


def index_matrix(num_of_cols, num_of_rows):
    idx_mat = []
    for i in range(num_of_rows):
        idx_mat.append([])
        for j in range(num_of_cols):
            idx_mat[i].append([i, j])
    return np.array(idx_mat)




# basic algorithm
def get_costs(energy_mat, isForward, c_l, c_v, c_r,rows,cols):
    cost_mat = np.copy(energy_mat)
    for i in range(1, rows):
        num_of_cols = cols
        m_prev_left = np.roll(cost_mat[i - 1], 1)
        m_prev_right = np.roll(cost_mat[i - 1], -1)
        m_prev_middle = np.copy(cost_mat[i - 1])
        m_left = cost_mat[i] + m_prev_left
        m_right = cost_mat[i] + m_prev_right
        m_middle = cost_mat[i] + m_prev_middle
        if isForward:
            m_left = m_left + c_l[i]
            m_right = m_right + c_r[i]
            m_middle = m_middle + c_v[i]
        cost_mat[i] = np.minimum(np.minimum(m_left, m_middle), m_right)
        if isForward:
            cost_mat[i][0] = energy_mat[i][0] + min(cost_mat[i - 1][0] + c_v[i][0], cost_mat[i - 1][1] + c_r[i][0])
            cost_mat[i][num_of_cols - 1] = energy_mat[i][num_of_cols - 1] + min(
                cost_mat[i - 1][num_of_cols - 1] + c_v[i][num_of_cols - 1],
                cost_mat[i - 1][num_of_cols - 2] + c_l[i][num_of_cols - 1])
        else:
            cost_mat[i][0] = energy_mat[i][0] + min(cost_mat[i - 1][0], cost_mat[i - 1][1])
            cost_mat[i][num_of_cols - 1] = energy_mat[i][num_of_cols - 1] + min(cost_mat[i - 1][num_of_cols - 1],
                                                                                cost_mat[i - 1][num_of_cols - 2])
    return cost_mat


def calculate_c_v(grayScale_mat,cols):
    gs_i_j_plus_1 = np.roll(grayScale_mat, -1, axis=1)  # cols
    #gs_i_j_plus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    gs_i_j_minus_1 = np.roll(grayScale_mat, 1, axis=1)  # cols
    #gs_i_j_minus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    c_v = abs(gs_i_j_plus_1 - gs_i_j_minus_1)
    c_v[:,cols-1]=255
    c_v[:,0]=255
    return c_v


def calculate_c_l(grayScale_mat,cols):
    gs_i_j_plus_1 = np.roll(grayScale_mat, -1, axis=1)  # cols
    #gs_i_j_plus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    gs_i_j_minus_1 = np.roll(grayScale_mat, 1, axis=1)  # cols
    #gs_i_j_minus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    gs_i_minus_1_j = np.roll(grayScale_mat, 1, axis=0)  # rows
    #gs_i_minus_1_j[-1, ...] = grayScale_mat[-2, ...]
    c_l = abs(gs_i_j_plus_1 - gs_i_j_minus_1) + abs(gs_i_minus_1_j - gs_i_j_minus_1)
    c_l[0] = abs(gs_i_j_plus_1[0]-gs_i_j_minus_1[0])+255
    c_l[:,cols-1]=255 + abs(gs_i_minus_1_j[:,cols-1]-gs_i_j_minus_1[:,cols-1])
    c_l[:,0]=255+255

    return c_l


def calculate_c_r(grayScale_mat,cols):
    gs_i_j_plus_1 = np.roll(grayScale_mat, -1, axis=1)  # cols
    #gs_i_j_plus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    gs_i_minus_1_j = np.roll(grayScale_mat, 1, axis=0)  # rows
    #gs_i_minus_1_j[-1, ...] = grayScale_mat[-2, ...]
    gs_i_j_minus_1 = np.roll(grayScale_mat, 1, axis=1)  # cols
    #gs_i_j_minus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    c_r = abs(gs_i_j_plus_1 - gs_i_j_minus_1) + abs(gs_i_minus_1_j - gs_i_j_plus_1)
    c_r[0] = abs(gs_i_j_plus_1[0]-gs_i_j_minus_1[0])+255
    c_r[:,cols-1]=255+255
    c_r[:,0]=255+abs(gs_i_minus_1_j[:,0]-gs_i_j_plus_1[:,0])

    return c_r


def backtracking(cost_mat, energy_mat, c_v, c_l, isForward,rows,cols):
    seam = []
    min_last_col = 0
    num_of_rows = rows
    min_last_val = cost_mat[num_of_rows - 1][0]
    for j in range(cols):
        if cost_mat[num_of_rows - 1][j] < min_last_val:
            min_last_col = j
            min_last_val = cost_mat[num_of_rows - 1][j]
    elem=[num_of_rows - 1, min_last_col]
    seam.append(elem)
    i = num_of_rows - 1
    while i > 0:
        if isForward:
            if cost_mat[i][min_last_col] == energy_mat[i][min_last_col] + cost_mat[i - 1][min_last_col] + \
                    c_v[i][min_last_col]:
                min_last_col = min_last_col
            elif cost_mat[i][min_last_col] == energy_mat[i][min_last_col] + cost_mat[i - 1][min_last_col - 1] + \
                    c_l[i][min_last_col]:
                min_last_col = min_last_col - 1
            else:
                min_last_col = min_last_col + 1
        else:
            if cost_mat[i][min_last_col] == energy_mat[i][min_last_col] + cost_mat[i - 1][min_last_col]:
                min_last_col = min_last_col
            elif cost_mat[i][min_last_col] == energy_mat[i][min_last_col] + cost_mat[i - 1][min_last_col - 1]:
                min_last_col = min_last_col - 1
            else:
                min_last_col = min_last_col + 1
        i = i - 1
        elem=[i, min_last_col]
        seam.append(elem)

    return seam


def seam_from_original_image(relative_seam, idx_mat,dict_col):
    original_seam = []
    for i in range(len(relative_seam)):
        x = relative_seam[i][0]
        y = relative_seam[i][1]
        original_x = idx_mat[x][y][0]
        original_y = idx_mat[x][y][1]
        original_seam.append([original_x, original_y])
        dict_col[original_x].add(original_y)
    return original_seam


# seam is ordered from last to first row
def remove_seam(mat, seam,rows,cols):
    seam = seam[::-1]# reversing the seam so that the first cell has the index from the first row
    for i in range(rows):
        j = seam[i][1]
        mat[i][ j:-1] = mat[i][ j + 1:]  # shift
        # mat.resize((rows, cols - 1))
        # mat = np.delete(mat, np.s_[-1:], axis=1)

# seam is ordered from last to first row
def remove_seam_from_idx(mat, seam,rows,cols):
    seam = seam[::-1]# reversing the seam so that the first cell has the index from the first row
    for i in range(rows):
        j = seam[i][1]
        mat[i][:, j:-1] = mat[i][:, j + 1:]  # shift
        # mat.resize((rows, cols - 1))
        # mat = np.delete(mat, np.s_[-1:], axis=1)


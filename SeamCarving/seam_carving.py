from typing import Dict, Any

import numpy as np

import utils
NDArray = Any

def resize_width(image: NDArray, out_height: int, out_width: int, forward_implementation: bool,red:bool):
    grayScale_mat = utils.to_grayscale(image)
    energy_mat = utils.get_gradients(grayScale_mat)
    in_rows = image.shape[0]
    in_cols = image.shape[1]
    idx_mat = utils.index_matrix(in_cols, in_rows)
    seam_db = []
    delta_h = out_height - in_rows
    delta_w = out_width - in_cols

    # seam_list=list()
    dict_col = dict()
    for i in range(in_rows):
        dict_col[i] = set()
    for i in range(abs(delta_w)):
        c_l = utils.calculate_c_l(grayScale_mat, in_cols - i)
        c_v = utils.calculate_c_v(grayScale_mat, in_cols - i)
        c_r = utils.calculate_c_r(grayScale_mat, in_cols - i)
        cost_mat = utils.get_costs(energy_mat, forward_implementation, c_l, c_v, c_r, rows=in_rows, cols=in_cols - i)
        seam = utils.backtracking(cost_mat, energy_mat, c_v, c_l, forward_implementation, in_rows, in_cols - i)
        original_idx_seam = utils.seam_from_original_image(seam, idx_mat, dict_col)
        seam_db.append(original_idx_seam)
        utils.remove_seam(grayScale_mat, seam, in_rows, in_cols - i)
        utils.remove_seam(energy_mat, seam, in_rows, in_cols - i)
        utils.remove_seam(idx_mat, seam, in_rows, in_cols - i)

    copy_image = image.copy()
    new_image = np.zeros((in_rows, out_width, 3), dtype=np.float32)
    # decreace
    if delta_w < 0:
        for i in range(in_rows):
            new_j = 0
            for j in range(in_cols):
                print(i, j)
                if j not in dict_col[i]:
                    new_image[i][new_j] = image[i][j]
                    new_j = new_j + 1
                else:
                    if red:
                        copy_image[i][j] = [255, 0, 0]
                    else:
                        copy_image[i][j] = [0, 0, 0]
    if delta_w > 0:
        for i in range(in_rows):
            new_j = 0
            for j in range(in_cols):
                print(i, j)
                if j in dict_col[i]:
                    new_image[i][new_j] = image[i][j]
                    new_image[i][new_j + 1] = image[i][j]
                    new_j = new_j + 2
                    if red:
                        copy_image[i][j] = [255, 0, 0]
                    else:
                        copy_image[i][j] = [0, 0, 0]
                else:
                    new_image[i][new_j] = image[i][j]
                    new_j = new_j + 1
    if delta_w==0:
        new_image=image

    d = dict()
    d["original_img"] = image
    d["vertical_seams_colored"] = copy_image
    d["resized"] = new_image
    return d

def rotate(image,CCW: bool):
    if CCW:
        return np.rot90(image, k=1, axes=(0,1))
    else:
        return np.rot90(image, k=-1, axes=(0, 1))


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    d_width = resize_width(image, out_height, out_width, forward_implementation,True)
    vertical_seams_colored = d_width["vertical_seams_colored"]
    resized_width = d_width["resized"]

    rotate_resized_width = rotate(resized_width,True)
    d_height = resize_width(rotate_resized_width, out_width, out_height, forward_implementation,False)
    resized_height = d_height["resized"]
    horizontal_seams_colored = d_height["vertical_seams_colored"]

    final_image = rotate(resized_height,False)
    horizontal_seams_colored = rotate(horizontal_seams_colored,False)

    d = dict()
    d["resized"] = final_image
    d["vertical_seams_colored"] = vertical_seams_colored
    d["horizontal_seams"] = horizontal_seams_colored
    return d




    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}




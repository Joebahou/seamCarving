from typing import Dict, Any
import numpy as np
import utils
NDArray = Any


def index_matrix(num_of_cols,num_of_rows):
    idx_mat =[]
    for i in range(num_of_rows):
        idx_mat.append([])
        for j in range(num_of_cols):
            idx_mat[i].append([i,j])
    return idx_mat

def grayScaling(image):
    return utils.to_grayscale(image)

#basic algorithm
def get_costs(energy_mat, isForward,c_l,c_v,c_r):
    cost_mat=np.copy(energy_mat)
    for i in range(1,len(energy_mat)):
            num_of_cols = len(cost_mat[i])
            m_prev_left=np.roll(cost_mat[i-1], -1)
            m_prev_right=np.roll(cost_mat[i-1], 1)
            m_prev_middle=np.copy(cost_mat[i-1])
            m_left=cost_mat[i]+m_prev_left
            m_right=cost_mat[i]+m_prev_right
            m_middle=cost_mat[i]+m_prev_middle
            if isForward:
                m_left = m_left + c_l[i]
                m_right = m_right+ c_r[i]
                m_middle = m_middle + c_v[i]
            cost_mat[i]=np.minimum(np.minimum(m_left,m_middle),m_right)
            if isForward:
                cost_mat[i][0] = energy_mat[i][0] + min(cost_mat[i - 1][0]+c_v[i][0], cost_mat[i - 1][1]+c_r[i][0])
                cost_mat[i][num_of_cols - 1] = energy_mat[i][num_of_cols - 1] + min(cost_mat[i - 1][num_of_cols - 1]+c_v[i][num_of_cols - 1],
                                                                                    cost_mat[i - 1][num_of_cols - 2]+c_l[i][num_of_cols - 1])
            else:
                cost_mat[i][0]=energy_mat[i][0]+min(cost_mat[i-1][0],cost_mat[i-1][1])
                cost_mat[i][num_of_cols-1]=energy_mat[i][num_of_cols-1]+min(cost_mat[i-1][num_of_cols-1],cost_mat[i-1][num_of_cols-2])
    return cost_mat

def calculate_c_v(grayScale_mat):
    gs_i_j_plus_1 = np.roll(grayScale_mat, -1, axis=1)  # cols
    gs_i_j_plus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    gs_i_j_minus_1 = np.roll(grayScale_mat, 1, axis=1)  # cols
    gs_i_j_minus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    c_v = abs(gs_i_j_plus_1 - gs_i_j_minus_1)
    return c_v

def calculate_c_l(grayScale_mat):
    gs_i_j_plus_1 = np.roll(grayScale_mat, -1, axis=1)  # cols
    gs_i_j_plus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    gs_i_j_minus_1 = np.roll(grayScale_mat, 1, axis=1)  # cols
    gs_i_j_minus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    gs_i_minus_1_j = np.roll(grayScale_mat, 1, axis=0)  # rows
    gs_i_minus_1_j[-1, ...] = grayScale_mat[-2, ...]
    c_l=abs(gs_i_j_plus_1-gs_i_j_minus_1)+abs(gs_i_minus_1_j-gs_i_j_minus_1)
    return c_l

def calculate_c_r(grayScale_mat):
    gs_i_j_plus_1 = np.roll(grayScale_mat, -1, axis=1)  # cols
    gs_i_j_plus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    gs_i_minus_1_j = np.roll(grayScale_mat, 1, axis=0)  # rows
    gs_i_minus_1_j[-1, ...] = grayScale_mat[-2, ...]
    gs_i_j_minus_1 = np.roll(grayScale_mat, 1, axis=1)  # cols
    gs_i_j_minus_1[:, -1, ...] = grayScale_mat[:, -2, ...]
    c_r=abs(gs_i_j_plus_1-gs_i_j_minus_1)+abs(gs_i_minus_1_j-gs_i_j_plus_1)
    return c_r

def backtracking(cost_mat,energy_mat,c_v,c_l,isForward):
    seam=[]
    min_last_col=0
    num_of_rows=len(cost_mat)
    min_last_val=cost_mat[num_of_rows-1][0]
    for j in range(len(cost_mat[0])):
        if(cost_mat[num_of_rows-1][j]<min_last_val):
            min_last_col=j
            min_last_val=cost_mat[num_of_rows-1][j]
    seam.append([num_of_rows-1,min_last_col])
    i =num_of_rows-1
    while i>0:
        if isForward:
            if cost_mat[i][min_last_col]==energy_mat[i][min_last_col]+cost_mat[i-1][min_last_col]+c_v[i][min_last_col]:
                min_last_col=min_last_col
            elif cost_mat[i][min_last_col]==energy_mat[i][min_last_col]+cost_mat[i-1][min_last_col-1]+c_l[i][min_last_col]:
                min_last_col=min_last_col-1
            else:
                min_last_col=min_last_col+1
        else:
            if cost_mat[i][min_last_col]==energy_mat[i][min_last_col]+cost_mat[i-1][min_last_col]:
                min_last_col=min_last_col
            elif cost_mat[i][min_last_col]==energy_mat[i][min_last_col]+cost_mat[i-1][min_last_col-1]:
                min_last_col=min_last_col-1
            else:
                min_last_col=min_last_col+1
        i=i-1
        seam.append([i,min_last_col])
    return seam




def seam_from_original_image(relative_seam,idx_mat):
    original_seam=[]
    for i in range(len(relative_seam)):
        x= relative_seam[i][0]
        y= relative_seam[i][1]
        original_x=idx_mat[x][y][0]
        original_y=idx_mat[x][y][1]
        original_seam.append([original_x,original_y])
    return original_seam

def remove_seam(mat,seam):
    for i in range(len(mat)):
        mat[:, j:-1] = mat[:, j+1:]
















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
    raise NotImplementedError('You need to implement this!')
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}



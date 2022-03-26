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

#basic algorithm
def get_costs(energy_mat):
    cost_mat=np.copy(energy_mat)
    for i in range(1,len(energy_mat)):
        num_of_cols = len(cost_mat[i])
        m_prev_left=np.roll(cost_mat[i-1], -1)
        m_prev_right=np.roll(cost_mat[i-1], 1)
        m_prev_middle=np.copy(cost_mat[i-1])
        m_left=cost_mat[i]+m_prev_left
        m_right=cost_mat[i]+m_prev_right
        m_middle=cost_mat[i]+m_prev_middle
        cost_mat[i]=np.minimum(np.minimum(m_left,m_middle),m_right)
        cost_mat[i][0]=energy_mat[i][0]+min(cost_mat[i-1][0],cost_mat[i-1][1])
        cost_mat[i][num_of_cols-1]=energy_mat[i][num_of_cols-1]+min(cost_mat[i-1][num_of_cols-1],cost_mat[i-1][num_of_cols-2])
    return cost_mat






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



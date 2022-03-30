from typing import Dict, Any

import numpy as np

import utils
NDArray = Any


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

    grayScale_mat=utils.to_grayscale(image)
    energy_mat=utils.get_gradients(grayScale_mat)
    in_rows=image.shape[0]
    in_cols=image.shape[1]
    idx_mat=utils.index_matrix(in_cols,in_rows)
    seam_db=[]
    delta_h=out_height-in_rows
    delta_w=out_width-in_cols
    c_l=utils.calculate_c_l(grayScale_mat)
    c_v=utils.calculate_c_v(grayScale_mat)
    c_r=utils.calculate_c_r(grayScale_mat)
    #seam_list=list()
    dict_col=dict()
    for i in range(in_rows):
        dict_col[i]=set()
    for i in range(abs(delta_w)):
        cost_mat=utils.get_costs(energy_mat,forward_implementation,c_l,c_v,c_r,rows=in_rows,cols=in_cols-i)
        seam=utils.backtracking(cost_mat,energy_mat,c_v,c_l,forward_implementation,in_rows,in_cols-i)
        original_idx_seam=utils.seam_from_original_image(seam,idx_mat,dict_col)
        seam_db.append(original_idx_seam)
        utils.remove_seam(grayScale_mat,seam,in_rows,in_cols-i)
        utils.remove_seam(energy_mat,seam,in_rows,in_cols-i)
        utils.remove_seam(idx_mat,seam,in_rows,in_cols-i)
    #decreace
    copy_image=image.copy()
    new_image=np.zeros((in_rows,out_width,3), dtype=np.float32)
    if delta_w<0:
        for i in range(in_rows):
            new_j=0
            for j in range(in_cols):
                print(i, j)
                if j not in dict_col[i]:
                    new_image[i][new_j]=image[i][j]
                    new_j=new_j+1
                else:
                    copy_image[i][j]=[255,0,0]

    d=dict()
    d["original_img"]=image
    d["vertical_seams_colored"]=copy_image
    d["resized"]=new_image
    return d






    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}




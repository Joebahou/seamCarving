import argparse

import nearest_neighbor
import seam_carving
import utils


def get_args():
    """A Helper function that defines the program arguments."""
    parser = argparse.ArgumentParser(description='Image resizing application, supporting multiple different resizing '
                                                 'methods including [Nearest neighbor interpolation, '
                                                  'Seam Carving, Seam Carving with Feed Forward Implementation.')
    parser.add_argument('--image_path', type=str, help='The input image path')
    parser.add_argument('--output_dir', type=str, help='The output directory')
    parser.add_argument('--height', type=int, help='The output image height size')
    parser.add_argument('--width', type=int, help='The output image width size')
    parser.add_argument('--resize_method', type=str, help='The resizing method. Supported methods are '
                                                          '[nearest_neighbor, seam_carving].')
    parser.add_argument('--use_forward_implementation', action='store_true',
                        help='If set and seam_carving is used as a resizing method, then the forward-looking '
                             'implementation is used.')
    parser.add_argument('--out_prefix', type=str, help='Output filename prefix.', default='out')
    args = parser.parse_args()
    return args


def main(args):
    """
    The program main function.
    :param args: the command-line input arguments.
    """
    image = utils.open_image(args.image_path)
    if args.resize_method == 'nearest_neighbor':
        output = nearest_neighbor.resize(image, args.height, args.width)
    elif args.resize_method == 'seam_carving':
        output = seam_carving.resize(image, args.height, args.width,
                                     forward_implementation=args.use_forward_implementation)
    else:
        raise ValueError(f'Resize method {args.resize_method} is not supported')
    utils.save_images(output, args.output_dir, args.out_prefix)


if __name__ == '__main__':
    args = get_args()
    main(args)

from argparse import ArgumentParser
import os

import numpy as np
import cv2


def main(input_dir: str, output_dir: str):
    in_dir = os.path.abspath(input_dir)
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        file_name, _ = os.path.splitext(file)
        in_file_path = os.path.join(in_dir, file)
        out_file_path = os.path.join(out_dir, ''.join([file_name, '.png']))

        array = np.load(str(in_file_path)).astype(np.float)
        if array.dtype != np.uint8 or array.min() < 0 or array.max() > 255:
            array_normalized = cv2.normalize(array, None, alpha=0, beta=255,
                                             norm_type=cv2.NORM_MINMAX)
            array_normalized = array_normalized.astype('uint8')
        else:
            array_normalized = array
        cv2.imwrite(out_file_path, array_normalized)
        print (f'{out_file_path}')


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Converts files from seprsco format to wav.')

    arg_parser.add_argument(
        '-i',
        '--input_dir',
        metavar='path/to/dir',
        type=str,
        help="Input directory path."
    )

    arg_parser.add_argument(
        '-o',
        '--output_dir',
        metavar='path/to/dir',
        type=str,
        help="Output directory path."
    )

    args = arg_parser.parse_args()
    main(args.input_dir, args.output_dir)

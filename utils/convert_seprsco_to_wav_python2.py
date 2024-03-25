"""Converts files from seprsco (saved as .pkl) format to wav.
Attention! This is Python 2 script. This script uses python 2 nesmdb package for conversion.
"""
from argparse import ArgumentParser
import os
import pickle

from nesmdb.convert import seprsco_to_wav
import numpy as np
from scipy.io.wavfile import write as wavwrite


PKL_EXT = '.pkl'
WAV_EXT = '.wav'


def main(input_dir, output_dir):
    """Converts files from seprsco format to wav.

    Args:
        input_dir: Str, input directory path.
        output_dir: Str, output directory path.
    """
    in_dir = os.path.abspath(input_dir)
    out_dir = os.path.abspath(output_dir)

    for file in os.listdir(in_dir):
        file_name, file_ext = os.path.splitext(file)
        if file_ext != PKL_EXT:
            continue
        in_file_path = os.path.join(in_dir, file)
        out_file_path = os.path.join(out_dir, ''.join([file_name, WAV_EXT]))

        with open(in_file_path, 'rb') as f:
            seprsco = pickle.load(f)
        wav = seprsco_to_wav(seprsco)
        wav *= 32767.
        wav = np.clip(wav, -32767., 32767.)
        wav = wav.astype(np.int16)

        wavwrite(out_file_path, 44100, wav)  # 44.1 kHz timing resolution
        print '%s' % file


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

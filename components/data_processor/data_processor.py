"""This module contains classes to manage data processing"""
from enum import Enum
import logging
from multiprocessing import Pool
from pathlib import Path
import pickle
import subprocess

import numpy as np

from components.common.configuration import DataProcessorConfiguration
from components.common.constants import LOGGER_NAME
from components.common.constants import NOISE_MAX_VAL
from components.common.constants import NOISE_MIN_VAL
from components.common.constants import SEPRSCO_DATASET_DOWNLOAD_LINK
from components.common.constants import PULSES12_MAX_VAL
from components.common.constants import PULSES12_MIN_VAL
from components.common.constants import TRIANGLE_MAX_VAL
from components.common.constants import TRIANGLE_MIN_VAL
from components.common.logger_manager import LoggerManager
from components.common.logger_manager import LogWritingClass


class OriginalDataFormat(Enum):
    """A class to define supported original data formats."""
    # - 'seprsco', Separated Score Format. Contains Note information only,
    #  for 4 instruments for N number for timestamps at 24Hz.
    SEPRSCO = 'seprsco'


class RepresentationFormat(Enum):
    """A class to define supported data representation formats."""
    ROLL = 'roll'


# TODO: Use pytorch transform.
class DataProcessor(LogWritingClass):
    """A class to manage data processing.

    Note:
        Forward processing: original -> scaled -> cut -> represented.
        Backward processing: represented -> scaled -> original.
    """
    def __init__(self, config: DataProcessorConfiguration, logger_manager: LoggerManager):
        LogWritingClass.__init__(self,
                                 LOGGER_NAME.format(class_name=self.__class__.__name__),
                                 logger_manager)
        self._replace_if_exists = config.replace_if_exists

        self._original_format = OriginalDataFormat.SEPRSCO
        self._representation = RepresentationFormat(config.representation)
        self._sample_len = config.sample_len
        self._cutting_step = config.cutting_step
        self._rows = config.rows

        self._log(logging.INFO, 'Initialized Data Processor.')

    def _download_sepsco_data(self, download_to_dir: Path):
        """Downloads and extracts data.

        Args:
            download_to_dir: Where to download.
        """
        if (
            self._replace_if_exists
            or not download_to_dir.exists() or not any(download_to_dir.iterdir())
        ):
            tmp_file = Path('./nesmdb24_seprsco.tar.gz')
            if not check_file_exists_and_not_empty(tmp_file):
                raise RuntimeError(f'Error downloading nesmdb24_seprsco.tar.gz '
                                f'from {SEPRSCO_DATASET_DOWNLOAD_LINK}.')
            subprocess.check_call(['tar', 'xvfz', str(tmp_file), '-C', './tmp'])
            subprocess.check_call(['mv', './tmp', str(download_to_dir.resolve())])
            subprocess.check_call(['rm', './tmp', '-r'])
            subprocess.check_call(['rm', 'nesmdb24_seprsco.tar.gz'])
            raise NotImplementedError('Link does not work at the moment. '
                                      'Download and extract manually.')

        for path in (download_to_dir / Path('train')).iterdir():
            if path.is_file():
                with open(path, 'rb') as bin_file:
                    rate, nsamps, seprsco = pickle.load(bin_file)
                    self._log(logging.INFO,
                              f'Downloaded sepsco data. Example: rate={rate}, '
                              f'nsamps={nsamps}, seprsco={seprsco.T}.')
            break

    def download_data(self, download_to_dir: Path):
        """Downloads data.

        Attributes:
            download_to_dir: A path to download to.
        """
        if self._original_format == OriginalDataFormat.SEPRSCO:
            self._download_sepsco_data(download_to_dir)
        else:
            raise ValueError(f'Unsupported original format: {self._original_format} '
                             f'in download_data method.')

    def convert_to_wav(self):
        raise NotImplementedError(f'Use Python 2 util.')

    # Forward

    def scale(self, original_data_path: Path, scaled_data_path: Path):
        """Scales data using min-max scaling according to instruments range."""
        if self._original_format == OriginalDataFormat.SEPRSCO:
            scaled_data_path.mkdir(parents=True, exist_ok=True)
            with Pool() as pool:
                pool.starmap(scale_song_min_max,
                             [(file, scaled_data_path)
                              for file in list(original_data_path.iterdir())])
        else:
            raise ValueError(f'Unsupported original format: {self._original_format} '
                             f'in scale method.')

    def cut(self, songs_dir: Path, cuts_dir: Path):
        """Cuts songs into peaces."""
        with Pool() as pool:
            cuts_dir.mkdir(parents=True, exist_ok=True)
            pool.starmap(cut_song, [(file, cuts_dir, self._sample_len, self._cutting_step)
                                    for file in list(songs_dir.iterdir())])

    def represent(self, songs_dir: Path, model_data_path: Path):
        with Pool() as pool:
            model_data_path.mkdir(parents=True, exist_ok=True)
            pool.starmap(represent_as_piano_roll_stack,
                         [(file, model_data_path, self._rows)
                          for file in list(songs_dir.iterdir())])

    # Backward

    def unrepresent(self, model_data_path: Path, songs_dir: Path):
        with Pool() as pool:
            songs_dir.mkdir(parents=True, exist_ok=True)
            pool.starmap(unrepresent_as_piano_roll_stack,
                         [(file, songs_dir, self._rows)
                          for file in list(model_data_path.iterdir())])

    def unscale(self, scaled_data_path: Path, result_data_path: Path):
        if self._original_format == OriginalDataFormat.SEPRSCO:
            result_data_path.mkdir(parents=True, exist_ok=True)
            with Pool() as pool:
                pool.starmap(unscale_song_min_max,
                             [(file, result_data_path)
                              for file in list(scaled_data_path.iterdir())])
        else:
            raise ValueError(f'Unsupported original format: {self._original_format} '
                             f'in unscale method.')


def check_file_exists_and_not_empty(file_path: Path):
    path = Path(file_path)
    return path.is_file() and path.stat().st_size > 0


def read_seprsco_song(song_path: Path):
    with open(song_path, 'rb') as f:
        _, _, seprsco = pickle.load(f)
    return seprsco.T


def save_seprsco_song(song_data: np.ndarray, song_path: Path):
    with open(song_path, 'wb') as f:
        pickle.dump(song_data, f, protocol=2)


def scale_instrument_min_max(inst_data: np.ndarray, inst_min: float, inst_max: float):
    inst_data[inst_data == 0] = inst_min
    inst_data -= inst_min
    inst_data /= (inst_max - inst_min)


def scale_song_min_max(input_path: Path, output_dir: Path):
    song = read_seprsco_song(input_path).astype(np.float)
    scale_instrument_min_max(song[:2, :], PULSES12_MIN_VAL, PULSES12_MAX_VAL)
    scale_instrument_min_max(song[2, :], TRIANGLE_MIN_VAL, TRIANGLE_MAX_VAL)
    scale_instrument_min_max(song[3, :], NOISE_MIN_VAL, NOISE_MAX_VAL)
    np.save(str(output_dir / input_path.stem), song)


def unscale_instrument_min_max(inst_data: np.ndarray, inst_min: float, inst_max: float):
    inst_data *= (inst_max - inst_min)
    inst_data += inst_min
    inst_data[inst_data == inst_min] = 0


def unscale_song_min_max(input_path: Path, output_dir: Path):
    song = np.load(str(input_path)).astype(np.float)

    unscale_instrument_min_max(song[:2, :], PULSES12_MIN_VAL, PULSES12_MAX_VAL)
    unscale_instrument_min_max(song[2, :], TRIANGLE_MIN_VAL, TRIANGLE_MAX_VAL)
    unscale_instrument_min_max(song[3, :], NOISE_MIN_VAL, NOISE_MAX_VAL)

    save_seprsco_song(song, output_dir / f'{input_path.stem}.pkl')


def cut_song(input_path: Path, output_dir: Path, sample_len: int, cutting_step: int):
    song = np.load(str(input_path)).astype(np.float)
    # print()
    # print('SONG', input_path, song.shape)
    for i, start in enumerate(range(0, len(song[0]) - sample_len + 1, cutting_step)):
        end = start + sample_len
        np.save(str(output_dir / Path(f'{input_path.stem}__{i}')), song[:, start:end])
    #     print('SAVED', str(output_dir / Path(f'{input_path.stem}__{i}')), '\n', song[:, start:end], '\n--')
    # print('===\n')


def represent_as_piano_roll_stack(input_path: Path, output_path: Path, rows: int):
    song = np.load(str(input_path)).astype(np.float)
    if song.shape[1] % rows != 0:
        raise ValueError(song.shape[1], rows)

    delta = song.shape[1] // rows
    stack_list = []
    for ri in range(1, rows + 1):
        stack_list.append(song[:, (ri-1) * delta : ri * delta])

    np.save(str(output_path / input_path.stem), np.stack(stack_list).reshape(4 * rows, delta, 1))


def unrepresent_as_piano_roll_stack(input_path: Path, output_path: Path, rows: int):
    song = np.load(str(input_path)).astype(np.float)
    stack_list = np.split(song, rows)
    np.save(str(output_path / input_path.stem), np.concatenate(stack_list, axis=1))

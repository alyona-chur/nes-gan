"""Downloads and prepares data for training."""
from argparse import ArgumentParser
from pathlib import Path
import shutil
import sys
from typing import Any, Dict

from omegaconf import OmegaConf

ROOT_DIR = str(Path(__file__).parents[0].resolve().parents[1].resolve())
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR)

from components.common.configuration import DataProcessorConfiguration
from components.common.constants import TMP_DATA_DIR
from components.common.logger_manager import LoggerManager
from components.common.path_lib import get_absolute_path
from components.data_processor.data_processor import DataProcessor


def main(configuration: Dict[Any, Any]):
    """Downloads and prepares data for training.

    Args:
        input_dir: Str, input directory path.
        output_dir: Str, output directory path.
    """
    logger = LoggerManager(None)
    config = DataProcessorConfiguration(configuration.data_processing)

    result_data_dir = get_absolute_path(config.data_dir, ROOT_DIR)
    downloaded_data_dir = result_data_dir / Path('nesmdb24_seprsco_united')
    represented_data_dir = result_data_dir / Path('training') / Path(
        f'nesmdb24_seprsco_{config.representation}_len{config.sample_len}_'
        f'step{config.cutting_step}_row{config.rows}')
    visible_data_dir = result_data_dir / Path('training') / Path(
        f'nesmdb24_seprsco_{config.representation}_len{config.sample_len}_'
        f'step{config.cutting_step}_row{config.rows}_visible')
    playable_data_dir = result_data_dir / Path('training') / Path(
        f'nesmdb24_seprsco_{config.representation}_len{config.sample_len}_'
        f'step{config.cutting_step}_row{config.rows}_playable')

    tmp_data_dir = get_absolute_path(TMP_DATA_DIR, ROOT_DIR)
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    scaled_data_dir = tmp_data_dir / Path('scaled')
    cut_data_dir = tmp_data_dir / Path('cut')

    data_processor = DataProcessor(config, logger)
    data_processor.cut(downloaded_data_dir, cut_data_dir)
    # data_processor.convert_to_wav(cut_data_dir, playable_data_dir)
    data_processor.scale(cut_data_dir, scaled_data_dir)
    data_processor.represent(scaled_data_dir, represented_data_dir)
    data_processor.convert_to_png(represented_data_dir, visible_data_dir)

    shutil.rmtree(tmp_data_dir)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Downloada and prepares data for training.')

    arg_parser.add_argument(
        '-c',
        '--config_path',
        metavar='path/to/dir',
        type=str,
        help="Configuration path.",
        default='./params.yaml'
    )

    parsed_args = arg_parser.parse_args()
    config = OmegaConf.load(parsed_args.config_path)
    main(config)

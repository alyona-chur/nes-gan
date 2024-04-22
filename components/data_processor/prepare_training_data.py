"""Prepares data for training."""
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
from components.data_processor.data_processor import downloaded_data_paths


def main(configuration: Dict[Any, Any]):
    """Prepares data for training.

    Args:
        configuration: Parsed configuration.
    """
    logger = LoggerManager(None)
    downloaded_data_path = get_absolute_path(configuration.downloaded_data_dir, ROOT_DIR)
    config = DataProcessorConfiguration(configuration.data_processor)

    (downloaded_united_data_dir,
     downloaded_train_data_dir, downloaded_valid_data_dir,
     downloaded_test_data_dir) = downloaded_data_paths(downloaded_data_path)

    result_data_dir = get_absolute_path(configuration.train_data_dir, ROOT_DIR)
    represented_united_data_dir = result_data_dir / Path('united')
    visible_united_data_dir = result_data_dir / Path('united_visible')
    playable_united_data_dir = result_data_dir / Path('united_playable')
    represented_train_data_dir = result_data_dir / Path('train')
    represented_valid_data_dir = result_data_dir / Path('valid')
    represented_test_data_dir = result_data_dir / Path('test')

    tmp_data_dir = get_absolute_path(TMP_DATA_DIR, ROOT_DIR)
    cut_data_dir = tmp_data_dir / Path('cut_train')
    scaled_data_dir = tmp_data_dir / Path('scaled_train')

    data_processor = DataProcessor(config, logger)

    # United
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    data_processor.cut(downloaded_united_data_dir, cut_data_dir)
    data_processor.convert_to_wav(cut_data_dir, playable_united_data_dir)
    data_processor.scale(cut_data_dir, scaled_data_dir)
    data_processor.represent(scaled_data_dir, represented_united_data_dir)
    data_processor.convert_to_png(represented_united_data_dir, visible_united_data_dir)
    shutil.rmtree(tmp_data_dir)

    # Train
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    data_processor.cut(downloaded_train_data_dir, cut_data_dir)
    data_processor.scale(cut_data_dir, scaled_data_dir)
    data_processor.represent(scaled_data_dir, represented_train_data_dir)
    shutil.rmtree(tmp_data_dir)

     # Valid
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    data_processor.cut(downloaded_valid_data_dir, cut_data_dir)
    data_processor.scale(cut_data_dir, scaled_data_dir)
    data_processor.represent(scaled_data_dir, represented_valid_data_dir)
    shutil.rmtree(tmp_data_dir)

    # Test
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    data_processor.cut(downloaded_test_data_dir, cut_data_dir)
    data_processor.scale(cut_data_dir, scaled_data_dir)
    data_processor.represent(scaled_data_dir, represented_test_data_dir)
    shutil.rmtree(tmp_data_dir)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Prepares data for training.')

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

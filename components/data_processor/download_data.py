"""Downloads data for training."""
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
    """Downloads data for training.

    Args:
        configuration: Parsed configuration.
    """
    logger = LoggerManager(None)
    config = DataProcessorConfiguration(configuration.data_processing)

    result_data_dir = get_absolute_path(config.data_dir, ROOT_DIR)
    downloaded_data_dir = result_data_dir / Path('nesmdb24_seprsco_united')

    data_processor = DataProcessor(config, logger)
    data_processor.download_data(downloaded_data_dir)


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Downloads data for training.')

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

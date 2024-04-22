"""Downloads data for training."""
from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Any, Dict

from omegaconf import OmegaConf

ROOT_DIR = str(Path(__file__).parents[0].resolve().parents[1].resolve())
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR)

from components.common.configuration import DataProcessorConfiguration
from components.common.logger_manager import LoggerManager
from components.common.path_lib import get_absolute_path
from components.data_processor.data_processor import DataProcessor
from components.data_processor.data_processor import downloaded_data_paths


def main(configuration: Dict[Any, Any]):
    """Downloads data for training.

    Args:
        configuration: Parsed configuration.
    """
    logger = LoggerManager(None)
    result_data_dir = get_absolute_path(configuration.downloaded_data_dir, ROOT_DIR)
    config = DataProcessorConfiguration(configuration.data_processor)

    (downloaded_united_data_dir,
     downloaded_train_data_dir, downloaded_valid_data_dir,
     downloaded_test_data_dir) = downloaded_data_paths(result_data_dir)

    data_processor = DataProcessor(config, logger)
    data_processor.download_data(downloaded_united_data_dir, downloaded_train_data_dir,
                                 downloaded_valid_data_dir, downloaded_test_data_dir)


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

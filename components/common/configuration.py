"""This module contains configuration for components."""
from pathlib import Path
from typing import Any, Dict

import yaml


class DataProcessorConfiguration:
    """A class to define Data Processor configuration."""
    def __init__(self, config_data: Dict[str, Any]):
        """Initializes and instance of the class.

        Args:
            config_data: Configuration.
        """
        self.representation = str(config_data['representation'])
        self.sample_len = int(config_data['sample_len'])
        self.cutting_step = int(config_data['cutting_step'])
        self.rows = int(config_data['rows'])

    def __repr__(self) -> str:
        return str(self.__dict__)


def read_yaml(config_path: Path) -> Dict[str, Any]:
    """Reads yaml file.

    Args:
        config_path: A path to yaml file.

    Returns:
        Content in dictionary format.
    """
    loader = yaml.FullLoader
    with open(config_path):
        return yaml.load(config_path, Loader=loader)

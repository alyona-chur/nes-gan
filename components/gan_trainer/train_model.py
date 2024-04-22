"""Trains GAN."""
from argparse import ArgumentParser
from pathlib import Path
import sys
import shutil
from typing import Any, Dict

import mlflow
from omegaconf import OmegaConf

ROOT_DIR = str(Path(__file__).parents[0].resolve().parents[1].resolve())
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR)

from components.common.constants import DEFAULT_LOGGING_DIR
from components.common.constants import GENERATED_DATA_DIR
from components.common.constants import RANDOM_SEED
from components.common.constants import TMP_GENERATED_DATA_DIR
from components.common.configuration import DataProcessorConfiguration
from components.common.logger_manager import LoggerManager
from components.common.path_lib import get_absolute_path
from components.data_processor.data_processor import DataProcessor
from components.gan_trainer.discriminator import get_discriminator
from components.gan_trainer.gan_trainer import GanTrainer
from components.gan_trainer.generator import get_generator


def main(configuration: Dict[Any, Any]):
    """Trains GAN.

    Args:
        configuration: Parsed configuration.
    """
    # Init Logging
    if configuration.gan_training.mlflow_server_url is not None:
        mlflow.set_tracking_url(configuration.gan_training.mlflow_server_url)
    log_file_name = configuration.gan_training.experiment_name.lower().replace(' ', '_')
    logger = LoggerManager(log_file_path=get_absolute_path(
        DEFAULT_LOGGING_DIR, ROOT_DIR) / Path(f'{log_file_name}.log'))

    # Init directories
    data_dir = get_absolute_path(configuration.train_data_dir, ROOT_DIR)
    train_data_dir, valid_data_dir = data_dir / Path('train'), data_dir / Path('valid')
    generated_data_dir = get_absolute_path(GENERATED_DATA_DIR, ROOT_DIR)
    cnt = 0
    while generated_data_dir.is_dir():
        generated_data_dir = get_absolute_path(f'{GENERATED_DATA_DIR}_{cnt}', ROOT_DIR)
        cnt += 1
    tmp_dir = get_absolute_path(TMP_GENERATED_DATA_DIR, ROOT_DIR)
    if tmp_dir.is_dir():
        shutil.rmtree(str(tmp_dir))

    # Init Models
    data_processor_config = DataProcessorConfiguration(configuration.data_processor)
    generator = get_generator(configuration.gan_training.generator_version,
                              configuration.gan_training.noise_vector_len,
                              data_processor_config.representation,
                              data_processor_config.sample_len,
                              data_processor_config.rows)
    discriminator = get_discriminator(configuration.gan_training.discriminator_version)
    trainer = GanTrainer(generator, configuration.gan_training.noise_vector_len,
                         discriminator, train_data_dir, valid_data_dir,
                         DataProcessor(data_processor_config, logger),
                         generated_data_dir, tmp_dir, logger)

    # Load Models
    if configuration.gan_training.generator_load_path is not None:
        trainer.load_generator_model(
            get_absolute_path(configuration.gan_training.generator_load_path, ROOT_DIR))
    if configuration.gan_training.discriminator_load_path is not None:
        trainer.load_discriminator_model(
            get_absolute_path(configuration.gan_training.discriminator_load_path, ROOT_DIR))

    # Train Models
    trainer.set_random_seeds(RANDOM_SEED)
    trainer.train_models(
        configuration.gan_training.experiment_name,
        configuration.gan_training.epochs,
        configuration.gan_training.batch_size,
        configuration.gan_training.generator_learning_rate,
        configuration.gan_training.discriminator_learning_rate,
        {
            "generator_version": configuration.gan_training.generator_version,
            "noise_vector_len": configuration.gan_training.noise_vector_len,
            "generator_load_path": configuration.gan_training.generator_load_path,
            "discriminator_version": configuration.gan_training.discriminator_version,
            "discriminator_load_path": configuration.gan_training.discriminator_load_path
        },
        configuration.gan_training.generator_steps_per_epoch,
        configuration.gan_training.discriminator_steps_per_epoch
    )


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='Trains GAN.')

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

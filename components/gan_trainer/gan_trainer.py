"""This module contains a GAN Trainer class."""
from abc import ABC
from io import BytesIO
from pathlib import Path
import shutil
from typing import Any, Dict, List

import cv2
import logging
import numpy as np
import mlflow
import tempfile
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from components.common.constants import GENERATIONS_NUM
from components.common.constants import LOGGER_NAME
from components.common.constants import SAVE_GENERATIONS_EACH_EPOCH_NUM
from components.common.constants import SAVE_MODELS_EACH_EPOCH_NUM
from components.common.logger_manager import LoggerManager
from components.common.logger_manager import LogWritingClass
from components.data_processor.data_processor import DataProcessor
from components.gan_trainer.generator import GeneratorBase


class NesmDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.file_list = list(data_dir.iterdir())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        file_path = self.file_list[idx]
        song = np.load(file_path)
        return torch.tensor(song, dtype=torch.float32).permute(2, 0, 1)


class GanTrainer(LogWritingClass):
    """Trains a Generative Adversarial Network (GAN) model.

    Attributes:
        _noise_vector_len: Int, a noise vector length for generator.
        _generator: nn.Module, a generator network of the GAN.
        _discriminator: nn.Module, a discriminator network of the GAN.
        _train_dataset: NesmDataset, dataset for training.
        _valid_dataset: NesmDataset, dataset for validation during training.
    """
    def __init__(self,
                 generator: nn.Module,
                 noise_vector_len: int,
                 discriminator: nn.Module,
                 train_data_path: Path,
                 valid_data_path: Path,
                 data_processor: DataProcessor,
                 tmp_dir: Path,
                 logger_manager: LoggerManager):
        LogWritingClass.__init__(self,
                                 LOGGER_NAME.format(class_name=self.__class__.__name__),
                                 logger_manager)
        # Models Architectures
        self._noise_vector_len: int = noise_vector_len
        self._generator: GeneratorBase = generator
        self._discriminator: nn.Module = discriminator

        # Data Loaders
        self._train_dataset = NesmDataset(train_data_path)
        self._valid_dataset = NesmDataset(valid_data_path)

        # Training
        self._data_processor = data_processor
        self._tmp_dir = tmp_dir
        self._train_dataloader: DataLoader = None
        self._valid_dataloader: DataLoader = None
        self._optimizer_G: optim.Adam = None
        self._optimizer_D: optim.Adam = None
        self._loss: nn.Module = None
        self._generator_steps_per_epoch = None
        self._discriminator_steps_per_epoch = None

    @property
    def generator(self) -> nn.Module:
        """Returns generator."""
        return self._generator

    @property
    def discriminator(self) -> nn.Module:
        """Returns discriminator."""
        return self._discriminator

    def _train_discriminator(self,
                             batch_size: int,
                             real_images: torch.Tensor,
                             real_disc_gt: torch.Tensor,
                             fake_disc_gt: torch.Tensor,
                             discriminator_gradient_norms: List[float]):
        noise = torch.randn(batch_size, self._noise_vector_len)
        fake_songs = self._generator(noise)

        self._optimizer_D.zero_grad()

        real_disc_predictions = self._discriminator(real_images)
        fake_disc_predictions = self._discriminator(fake_songs.detach())  # Avoid generator's grads
        # print(f'{real_disc_predictions.shape=}, {fake_disc_predictions.shape=}')
        d_loss = self._loss(real_disc_predictions, real_disc_gt) + self._loss(
                            fake_disc_predictions, fake_disc_gt)  # / 2.0 in original paper
        d_loss.backward()
        discriminator_gradient_norms.append(
            torch.nn.utils.clip_grad_norm_(self._discriminator.parameters(),
                                            max_norm=1000, norm_type=2).item())
        self._optimizer_D.step()
        return d_loss

    def _train_generator(self,
                         batch_size: int,
                         real_disc_gt: torch.Tensor,
                         generator_gradient_norms: List[float]):
        noise = torch.randn(batch_size, self._noise_vector_len)
        fake_songs = self._generator(noise)

        self._optimizer_G.zero_grad()

        # Generator wants discriminator to output 1 for fakes
        fake_disc_predictions_with_grads = self._discriminator(fake_songs)
        g_loss = self._loss(fake_disc_predictions_with_grads, real_disc_gt)
        g_loss.backward()
        generator_gradient_norms.append(
            torch.nn.utils.clip_grad_norm_(self._generator.parameters(),
                                            max_norm=1000, norm_type=2).item())
        self._optimizer_G.step()
        return g_loss

    def _train_epoch(self):
        generator_gradient_norms = []
        discriminator_gradient_norms = []

        for _, real_images in enumerate(self._train_dataloader):
            batch_size = real_images.size(0)
            real_disc_gt = torch.ones(batch_size, 1)
            fake_disc_gt = torch.zeros(batch_size, 1)

            for _ in range(self._discriminator_steps_per_epoch):
                last_d_loss = self._train_discriminator(batch_size, real_images, real_disc_gt,
                                                        fake_disc_gt, discriminator_gradient_norms)
            for _ in range(self._generator_steps_per_epoch):
                last_g_loss = self._train_generator(batch_size, real_disc_gt,
                                                    generator_gradient_norms)

        avg_discriminator_gradient_norm = sum(
            discriminator_gradient_norms) / len(discriminator_gradient_norms)
        avg_generator_gradient_norm = sum(
            generator_gradient_norms) / len(generator_gradient_norms)
        return (last_d_loss, last_g_loss,
                avg_discriminator_gradient_norm, avg_generator_gradient_norm)

    def _train(self, epochs: int,):
        for epoch in range(epochs):
            self._generator.train()
            self._discriminator.train()
            (last_d_loss, last_g_loss, avg_discriminator_gradient_norm,
             avg_generator_gradient_norm) = self._train_epoch()

            # Evaluation
            # TODO: Move to a separated method.
            self._generator.eval()
            self._discriminator.eval()
            if (epoch + 1) % SAVE_MODELS_EACH_EPOCH_NUM == 0 or epoch + 1 == epochs:
                mlflow.pytorch.log_model(self._generator, f'generator_epoch_{epoch}')
                mlflow.pytorch.log_model(self._discriminator, f'discriminator_epoch_{epoch}')

            with torch.no_grad():
                val_d_losses, val_g_losses = [], []
                for val_images in self._valid_dataloader:
                    batch_size = val_images.size(0)
                    noise = torch.randn(batch_size, self._noise_vector_len)
                    fake_images = self._generator(noise)

                    real_disc_gt = torch.ones(batch_size, 1)
                    fake_disc_gt = torch.zeros(batch_size, 1)
                    real_disc_predictions = self._discriminator(val_images)
                    fake_disc_predictions = self._discriminator(fake_images)

                    d_loss_val = self._loss(real_disc_predictions, real_disc_gt) + self._loss(
                        fake_disc_predictions, fake_disc_gt)
                    g_loss_val = self._loss(fake_disc_predictions, real_disc_gt)

                    val_d_losses.append(d_loss_val.item())
                    val_g_losses.append(g_loss_val.item())

                avg_d_loss_val = sum(val_d_losses) / len(val_d_losses)
                avg_g_loss_val = sum(val_g_losses) / len(val_g_losses)

                if (epoch + 1) % SAVE_GENERATIONS_EACH_EPOCH_NUM == 0 or epoch + 1 == epochs:
                    noise = torch.randn(GENERATIONS_NUM, self._noise_vector_len)
                    represented_songs_batch = self._generator(noise).detach().cpu()

                    self._tmp_dir.mkdir(parents=True, exist_ok=False)
                    represented_dir = self._tmp_dir / Path('represented')
                    represented_dir.mkdir(parents=True, exist_ok=False)
                    for i, represented_song in enumerate(represented_songs_batch):
                        represented_song = represented_song.permute(1, 2, 0).numpy()
                        png_song = cv2.normalize(
                            represented_song, None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX).astype('uint8').reshape((32, 32))

                        # Save .png directly
                        _, buffer = cv2.imencode('.png', png_song)
                        io_buffer = BytesIO(buffer)
                        with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as tmp:
                            io_buffer.seek(0)
                            tmp.write(io_buffer.read())
                            tmp_path = tmp.name
                            mlflow.log_artifact(tmp_path, artifact_path=f'results_epoch{epoch}_png')

                        # Save for wav
                        represented_file_path = represented_dir / Path(f'{i}.npy')
                        np.save(str(represented_file_path), represented_song)

                    self._data_processor.unrepresent(represented_dir,
                                                     self._tmp_dir / Path('scaled'))
                    self._data_processor.unscale(self._tmp_dir / Path('scaled'),
                                                 self._tmp_dir / Path('seprsco'))
                    self._data_processor.convert_to_wav(self._tmp_dir / Path('seprsco'),
                                                        self._tmp_dir / Path('wavs'))
                    for wav_file in (self._tmp_dir / Path('wavs')).iterdir():
                        mlflow.log_artifact(wav_file, artifact_path=f'results_epoch{epoch}_wav')

                    shutil.rmtree(str(self._tmp_dir))

            d_weights = torch.cat([p.view(-1) for p in self._discriminator.parameters()])
            g_weights = torch.cat([p.view(-1) for p in self._generator.parameters()])

            mlflow.log_metric('Discriminator LR', self._optimizer_D.param_groups[0]['lr'], step=epoch)
            mlflow.log_metric('Generator LR', self._optimizer_G.param_groups[0]['lr'], step=epoch)
            mlflow.log_metric('D Train Loss', last_d_loss.item(), step=epoch)
            mlflow.log_metric('G Train Loss', last_g_loss.item(), step=epoch)
            mlflow.log_metric('D Test Loss', avg_d_loss_val, step=epoch)
            mlflow.log_metric('G Test Loss', avg_g_loss_val, step=epoch)
            mlflow.log_metric('Average Discriminator Gradient Norm',
                              avg_discriminator_gradient_norm, step=epoch)
            mlflow.log_metric('Average Generator Gradient Norm',
                              avg_generator_gradient_norm, step=epoch)
            mlflow.log_metric('D Weight Mean', d_weights.mean().item(), step=epoch)
            mlflow.log_metric('G Weight Mean', g_weights.mean().item(), step=epoch)
            mlflow.log_metric('D Weight Std', d_weights.std().item(), step=epoch)
            mlflow.log_metric('G Weight Std', g_weights.std().item(), step=epoch)

            self._log(
                logging.INFO,
                f'Epoch [{epoch}/{epochs}], '
                f'Discriminator LR: {self._optimizer_D.param_groups[0]["lr"]:.4f}, '
                f'Generator LR: {self._optimizer_G.param_groups[0]["lr"]:.4f}, '
                f'D Train Loss: {last_d_loss.item():.4f}, '
                f'G Train Loss: {last_g_loss.item():.4f}, '
                f'D Test Loss: {avg_d_loss_val:.4f}, '
                f'G Test Loss: {avg_g_loss_val:.4f}, '
                f'Average Discriminator Gradient Norm: {avg_discriminator_gradient_norm:.4f}, '
                f'Average Generator Gradient Norm: {avg_generator_gradient_norm:.4f}, '
                f'D Weight Mean: {d_weights.mean().item():.4f}, '
                f'G Weight Mean: {g_weights.mean().item():.4f}, '
                f'D Weight Std: {d_weights.std().item():.4f}, '
                f'G Weight Std: {g_weights.std().item():.4f}.'
            )

    def _reset_training(self):
        self._train_dataloader = None
        self._valid_dataloader = None
        self._optimizer_G = None
        self._optimizer_D = None
        self._loss = None
        self._generator_steps_per_epoch = None
        self._discriminator_steps_per_epoch = None

    def set_random_seeds(self, random_seed: int):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_generator_model(self, model_path: Path):
        # TODO: Use MLFlow.
        self._generator = mlflow.pytorch.load_model(model_path)

    def load_discriminator_model(self, model_path: Path):
        # TODO: Use MLFlow.
        self._discriminator = mlflow.pytorch.load_model(model_path)

    def train_models(self,
                    mlflow_experiment_name: str,
                    epochs: int,
                    batch_size: int,
                    generator_learning_rate: float,
                    discriminator_learning_rate: float,
                    other_params_to_log: Dict[str, Any],
                    generator_steps_per_epoch: int = 1,
                    discriminator_steps_per_epoch: int = 1):
        self._reset_training()
        assert generator_steps_per_epoch >= 1 and discriminator_steps_per_epoch >= 1

        mlflow.set_experiment(mlflow_experiment_name)
        self._log(logging.INFO,
                  f'Starting training for {mlflow_experiment_name} experiment. '
                  f'Using device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')

        with mlflow.start_run():
            for param_name, param_value in other_params_to_log.items():
                mlflow.log_param(param_name, param_value)
            mlflow.log_param('data_representation', self._generator.data_representation_format)
            mlflow.log_param('data_noise_len', self._generator.noise_len)
            mlflow.log_param('data_sample_len', self._generator.sample_len)
            mlflow.log_param('data_rows', self._generator.rows)

            mlflow.log_param('epochs', epochs)
            mlflow.log_param('batch_size', batch_size)
            mlflow.log_param('generator_start_learning_rate', generator_learning_rate)
            mlflow.log_param('discriminator_start_learning_rate', discriminator_learning_rate)
            mlflow.log_param('generator_steps_per_epoch', generator_steps_per_epoch)
            mlflow.log_param('discriminator_steps_per_epoch', discriminator_steps_per_epoch)

            self._train_dataloader = DataLoader(self._train_dataset,
                                                batch_size=batch_size, shuffle=True)
            self._valid_dataloader = DataLoader(self._valid_dataset,
                                                batch_size=batch_size, shuffle=True)
            self._optimizer_G = optim.Adam(self._generator.parameters(),
                                           lr=generator_learning_rate)
            self._optimizer_D = optim.Adam(self._discriminator.parameters(),
                                           lr=discriminator_learning_rate)
            self._loss = nn.BCELoss()
            self._generator_steps_per_epoch = generator_steps_per_epoch
            self._discriminator_steps_per_epoch = discriminator_steps_per_epoch
            self._train(epochs)

        self._log(logging.INFO, f'Finished training for {mlflow_experiment_name} experiment.')

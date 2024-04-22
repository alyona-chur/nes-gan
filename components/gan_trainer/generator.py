"""This module contains Generator model definitions."""
from abc import ABC
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorVersion(Enum):
    """Enum for Generator versions."""
    VER0_LEN256_ROW8 = 'ver0_len256_row8'
    VER0_DEEPER_LEN256_ROW8 = 'ver0_deeper_len256_row8'
    VER0_DEEPER_LEN256_ROW1 = 'ver0_deeper_len256_row1'


class GeneratorBase(nn.Module, ABC):
    """A base class for Generator models."""
    def __init__(self,
                 version_name: str,
                 noise_len: int,
                 data_representation_format: str,
                 sample_len: int,
                 rows: int):
        ABC.__init__(self)
        nn.Module.__init__(self)

        self._version_name = version_name
        self._noise_len = noise_len

        self._data_representation_format = data_representation_format
        self._sample_len = sample_len
        self._rows = rows

    @property
    def version(self) -> str:
        return self._version_name

    @property
    def noise_len(self) -> int:
        return self._noise_len

    @property
    def data_representation_format(self) -> str:
        return self._data_representation_format

    @property
    def sample_len(self) -> str:
        return self._sample_len

    @property
    def rows(self) -> str:
        return self._rows


class Generator0Len256Row8(GeneratorBase):
    """Generator version model for GAN with [BATCH_SIZE, 1x32x32] output."""
    def __init__(self,
                 version_name: str,
                 noise_len: int,
                 data_representation_format: str,
                 sample_len: int,
                 rows: int):
        """Initializes tan instance of the class."""
        GeneratorBase.__init__(self, version_name, noise_len,
                               data_representation_format, sample_len, rows)
        self.fc = nn.Linear(noise_len, 128 * 8 * 8, bias=False)
        self.conv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, stride=1, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2, padding=1, bias=True)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (tensor): The input noise vector.

        Returns:
            tensor: The output image tensor in the range [0, 1].
        """
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)
        x = F.leaky_relu(self.batchnorm1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)), 0.2)
        x = torch.sigmoid(self.conv3(x))
        return x


class Generator0DeeperLen256Row8(GeneratorBase):
    """Generator version model for GAN with [BATCH_SIZE, 1x32x32] output."""
    def __init__(self,
                 version_name: str,
                 noise_len: int,
                 data_representation_format: str,
                 sample_len: int,
                 rows: int):
        """Initializes tan instance of the class."""
        GeneratorBase.__init__(self, version_name, noise_len,
                               data_representation_format, sample_len, rows)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_len, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64 x 8 x 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 32 x 16 x 16
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Sigmoid() # Output size: 1 x 32 x 32 - Error training the second model (nn.Tanh())!
        )

    def forward(self, z):
        z = z.view(-1, self._noise_len, 1, 1)
        return self.model(z)


class Generator0DeeperLen256Row1(GeneratorBase):
    """Generator version model for GAN with [BATCH_SIZE, 1x32x32] output."""
    def __init__(self,
                 version_name: str,
                 noise_len: int,
                 data_representation_format: str,
                 sample_len: int,
                 rows: int):
        """Initializes tan instance of the class."""
        GeneratorBase.__init__(self, version_name, noise_len,
                               data_representation_format, sample_len, rows)

        self.model = nn.Sequential(
            # Input: Latent vector Z of size [batch_size, noise_len, 1, 1]
            nn.ConvTranspose2d(noise_len, 512, (4, 4), 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Size: [batch_size, 512, 4, 4]
            nn.ConvTranspose2d(512, 256, (8, 3), (2, 1), (2, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Size: [batch_size, 256, 16, 4]
            nn.ConvTranspose2d(256, 128, (8, 1), (2, 1), (2, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Size: [batch_size, 128, 64, 4]
            nn.ConvTranspose2d(128, 64, (8, 1), (2, 1), (2, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Size: [batch_size, 64, 128, 4]
            nn.ConvTranspose2d(64, 32, (8, 1), (2, 1), (2, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # Size: [batch_size, 32, 256, 4]
            nn.ConvTranspose2d(32, 1, (1, 1), 1, 0, bias=False),
            nn.Sigmoid()  # Output size: [batch_size, 1, 256, 4]
        )

    def forward(self, z):
        z = z.view(-1, self._noise_len, 1, 1)
        return self.model(z)


def get_generator(ver: str,
                  noise_len: int,
                  data_representation_format: str,
                  sample_len: int,
                  rows: int):
    """Factory method to retrieve the generator model based on the version specified.

    Returns:
        nn.Module: Generator model.

    Raises:
        ValueError: If the specified version is not supported.
    """
    generator_ver = GeneratorVersion(ver)
    if generator_ver == GeneratorVersion.VER0_LEN256_ROW8:
        return Generator0Len256Row8(ver, noise_len, data_representation_format, sample_len, rows)
    if generator_ver == GeneratorVersion.VER0_DEEPER_LEN256_ROW8:
        return Generator0DeeperLen256Row8(ver, noise_len,
                                          data_representation_format, sample_len, rows)
    if generator_ver == GeneratorVersion.VER0_DEEPER_LEN256_ROW1:
        return Generator0DeeperLen256Row1(ver, noise_len,
                                          data_representation_format, sample_len, rows)
    raise ValueError(f'Unknown generator version: {ver}.')

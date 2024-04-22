"""This module contains Discriminator model definitions."""
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid


class DiscriminatorVersion(Enum):
    """Enum for Discriminator versions."""
    VER0 = 'ver0_len256_row8'


# TODO: Make a base class with properties like generator.py.
class Discriminator0Len256Row8(nn.Module):
    """Discriminator model for GAN."""
    def __init__(self):
        """Initializes an instance of the class."""
        super(Discriminator0Len256Row8, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 4 * 4, 1)  # torch.Size([BATCH_SIZE, 1])

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (tensor): The input data.

        Returns:
            tensor: The output data after applying convolutional and linear layers with activations.
        """
        x = F.leaky_relu(self.batchnorm1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)), 0.2)
        x = x.view(-1, 128 * 4 * 4)
        x = sigmoid(self.fc(x))
        return x


def get_discriminator(ver: str):
    """Factory method to retrieve the discriminator model based on the version specified.

    Args:
        ver (str): Version of the discriminator.

    Returns:
        nn.Module: Discriminator model.

    Raises:
        ValueError: If the specified version is not supported.
    """
    generator_ver = DiscriminatorVersion(ver)
    if generator_ver == DiscriminatorVersion.VER0:
        return Discriminator0Len256Row8()
    raise ValueError(f'Unknown discriminator version: {ver}.')

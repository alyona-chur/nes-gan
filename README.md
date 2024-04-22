# Generative Adversarial Network for Nintendo Entertainment System Music

## About

This project uses a Generative Adversarial Network (GAN) to create new 8-bit style music, trained on the [The NES Music Database](https://github.com/chrisdonahue/nesmdb). You can explore generated samples FIXME: [here](https://colab.research.google.com/github/alyona-chur/nesm-gan/blob/main/gan_for_nes_music_results_0.ipyn).

The development of this project incorporates some concepts from the '[Udemy End-to-End Machine Learning Course](https://www.udemy.com/course/sustainable-and-scalable-machine-learning-project-development/)', along with other resources.

## Usage

Follow these instructions to use the project:

1. **Setting Up the Environment.**
    - Check CUDA and cuDNN versions in '''docker/gpu.Dockerfile''' if using a GPU.
    - Build the Docker container by running:

```
./docker_build.sh -t [cpu|gpu]
```

2. **Downloading, Preparing Data, and Training Models.**

```
./docker_run_data_preparation_and_training.sh -t [cpu|gpu]
```

Generated samples are saved to '''./data/samples''' during training.

3. TODO: To generate music...

### Tested Configurations

Tested configurations include:

- GPU:
    Ubuntu 20.04
    Docker version 20.10.7
    Driver Version: 535.171.04
    CUDA Version: 12.2
- CPU:
    Ubuntu 20.04
    Docker version 24.0.7

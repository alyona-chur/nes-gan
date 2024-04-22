# Generative Adversarial Network for Nintendo Entertainment System Music

## About

This project uses a Generative Adversarial Network (GAN) to create new 8-bit style music, trained on the The NES Music Database. You can explore samples FIXME: [here](https://colab.research.google.com/github/alyona-chur/nesm-gan/blob/main/gan_for_nes_music_results_0.ipyn).

It also serves as a practical application of some concepts from the '[Udemy End-to-End Machine Learning: From Idea to Implementation Course](https://www.udemy.com/course/sustainable-and-scalable-machine-learning-project-development/)', among other resources.

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

3. TODO: To generate music with the latest available model...

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

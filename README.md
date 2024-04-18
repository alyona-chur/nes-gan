# Generative Adversarial Network for Nintendo Entertainment System Music

## About

This project generates new 8-bit style music using a Generative Adversarial Network (GAN).

## Usage

- To download, prepare data and train models from scratch run TODO.
- To run generation with the latest trained Generator model run TODO.
- To hear the lastest samples open `gan_for_nes_music_demo.ipynb` notebook in Google Colab.

## Data

This project uses songs in separated score format from [The NES Music Database](https://github.com/chrisdonahue/nesmdb).

### Sepsco Format

The NES synthesizer has five instrument voices: two pulse-wave generators (P1, P2), a triangle-wave generator (TR), a percussive noise generator (NO), and an audio sample playback channel (excluded for simplicity).

A score format is a piano roll representation. It is sampled at a fixed rate of 24. The separated score format contains note information that the NES synthesizer needs to render the music at each timestep. Each song is a numpy.uint8 array of size Nx4. This format is convenient for modeling the notes/timing of the music but misses expressive performance characteristics (velocity and timbre information).

### Trainig data

In this project for training used data that are in stacked piano roll-like format an array with NxM where N - is 4, M - is sample len (timestamps) and each value is a note plays by n instrument in m moment. At the end data is stacked to make more square-like matrix. Examples for sample len 256 (~10 sec) and number of rows 8 are the following.

PIC1 PIC2 PIC3

---

ML model building using DVC, Mlflow workflow for en-to-end development.
This project generates new 8-bit style music using a Generative Adversarial Network (GAN), trained on the [The NES Music Database](https://github.com/chrisdonahue/nesmdb).


The MLflow TensorFlow Guide is an educational project. This project demonstrates how to build, train, and manage a TensorFlow machine learning model using MLflow, a powerful open-source platform for the end-to-end machine learning lifecycle.

## E2E Machine Leanring

This is an edu / experimint project demonstarting a full ml / deep learning / gen ai lifecycle with nesm gan lstm music generation
dvc data versioning and data preparation pipeline
ml flow for model versioning and project packing
fast api + torch serve for service with generator in it with ability to have several model instances and redis queue

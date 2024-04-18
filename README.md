# Generative Adversarial Network for Nintendo Entertainment System Music

## About

This project generates new 8-bit style music using a Generative Adversarial Network (GAN), trained on the [The NES Music Database](https://github.com/chrisdonahue/nesmdb).

## Progress
- Model is in training. Check `gan_for_nes_music_results_*.ipynb` for latest music samples.

## Data - Seprsco Format
Trained on [The NES Music Database](https://github.com/chrisdonahue/nesmdb) in `nesmdb24_seprsco` format.

From WIKI:

sepsco format...


The NES synthesizer has five instrument voices: two pulse-wave generators (P1, P2), a triangle-wave generator (TR), a percussive noise generator (NO), and an audio sample playback channel (excluded for simplicity).

A score format is a piano roll representation. It is sampled at a fixed rate of 24.

The separated score format contains note information that the NES synthesizer needs to render the music at each timestep. Each song is a numpy.uint8 array of size Nx4. This format is convenient for modeling the notes/timing of the music but misses expressive performance characteristics (velocity and timbre information).

The following table displays the possible values for each of the instrument voices.


In this project data processing creates an array with NxM where N - is 4, M - is sample len (timestamps) and each value is a note plays by n instrument in m moment. At the end data is stacked to make more



Alternative: using wav + Furey

## Model

Models can be found here

## Usage
- To download data, train and eval from scratch run. Result: model, metrics, music samples.
- To run with trained models.
- To hear samples.
- Open `gan_for_nes_music_results_*.ipynb` to listen to generated music.

---

ML model building using DVC, Mlflow workflow for en-to-end development.
This project generates new 8-bit style music using a Generative Adversarial Network (GAN), trained on the [The NES Music Database](https://github.com/chrisdonahue/nesmdb).


The MLflow TensorFlow Guide is an educational project. This project demonstrates how to build, train, and manage a TensorFlow machine learning model using MLflow, a powerful open-source platform for the end-to-end machine learning lifecycle.


An edu / experimint project demonstarting a full ml / deep learning / gen ai lifecycle with nesm gan lstm music generation
dvc data versioning and data preparation pipeline
ml flow for model versioning and project packing
fast api + torch serve for service with generator in it with ability to have several model instances and redis queue

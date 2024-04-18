# DEV MANUAL

Docker version 20.10.7
Driver Version: 535.171.04
CUDA Version: 12.2

## Utils

- **convert_seprsco_to_wav_python2.py** is a tool for converting data to playable format (.wav) using original lib. *Note:* uses Python 2 because original [nesmbd](https://github.com/chrisdonahue/nesmdb) is a Python 2 module.

## Using GPU

- When using gpu, set the correct cuda and cudnn versions in gpu.Dockerfile.

## Jupyter Notebook in docker container

- To run jupyter notebook in docker container the following command. *Note:* port 10171 is the one that is exposed in this project by default.
```
jupyter notebook --no-browser --port=10172 --ip=0.0.0.0
```

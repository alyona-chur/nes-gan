#!/bin/bash

# Parse input parameters
IMAGE_TYPE="cpu"
PORT=10172
while getopts ":t:p:" opt; do
  case $opt in
    t) IMAGE_TYPE=$OPTARG;;
    p) PORT=$OPTARG;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# Set image type: cpu, gpu
GPU_CMD=''
if [ "$IMAGE_TYPE" == "gpu" ]; then
  IMAGE_NAME='nesm-gan-gpu'
  GPU_CMD='--gpus all '
elif [ "$IMAGE_TYPE" == "cpu" ]; then
  IMAGE_NAME='nesm-gan-cpu'
else
  echo "Unable to run docker container, unknown image type $IMAGE_TYPE"
  exit 1
fi

# Build command and run
cmd="docker run "
cmd+="-u $(id -u):$(id -g) "
cmd+="${GPU_CMD}"
cmd+="--mount type=bind,source=$(pwd),destination=/usr/src/app "
cmd+="-it "
cmd+="--rm "
cmd+="--name ${IMAGE_NAME}-$(whoami) "
cmd+="-p $PORT:$PORT "
cmd+="${IMAGE_NAME} "
cmd+="/bin/bash -c "
cmd+="\"python3.7 components/data_processor/download_data.py && "
cmd+="python3.7 components/data_processor/prepare_training_data.py && "
cmd+="python3.7 components/gan_trainer/train_model.py\""
# cmd+="/bin/bash"

echo "$cmd" && \
eval "$cmd"

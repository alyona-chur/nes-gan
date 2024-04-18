#!/bin/bash
# This script builds docker image.

# Parse input parameters
# -t for gpu usage
IMAGE_TYPE="cpu"
while getopts ":t:" opt; do
  case $opt in
    t) IMAGE_TYPE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2;;
  esac
done

# Set image type: cpu, gpu
if [ "$IMAGE_TYPE" == "gpu" ]; then
  IMAGE_NAME='nesm-gan-gpu'
  DOCKERFILE_PATH='docker/gpu.Dockerfile'
elif [ "$IMAGE_TYPE" == "cpu" ]; then
  IMAGE_NAME='nesm-gan-cpu'
  DOCKERFILE_PATH='docker/cpu.Dockerfile'
else
  echo "Unable to build docker image, unknown image type $IMAGE_TYPE";
  exit 1
fi

# Build command and run
cmd="docker build "
cmd+="--build-arg USER_NAME=$(whoami) "
cmd+="--build-arg USER_ID=$(id -u ${USER}) "
cmd+="--build-arg GROUP_ID=$(id -g ${USER}) "
cmd+="-t ${IMAGE_NAME} "
cmd+="-f ${DOCKERFILE_PATH} "
cmd+="."

echo "$cmd" && \
eval "$cmd"

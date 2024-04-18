#!/bin/bash
# This script runs docker in interactive way.

# Parse input parameters
# -t for gpu usage
IMAGE_TYPE="cpu"
while getopts ":t" opt; do
  case $opt in
    t) image_type="$OPTARG"
      if [ "$docker_image_type" != "" ]; then
        IMAGE_TYPE="$image_type"
      fi
    ;;
    \?) echo "Invalid option -$OPTARG" >&2;;
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
  echo "Unable to run docker container, unknown image type $IMAGE_TYPE";
  exit 1
fi

# Build command and run
cmd="docker run "
cmd+="-u $(id -u):$(id -g) "
cmd+="${GPU_CMD}"
cmd+="--mount type=bind,source=$(pwd),destination=/usr/src/app "
cmd+="-it "
cmd+="--rm "
cmd+="--name nesm-gan-$(whoami) "
cmd+="-p 10172:10172 "
cmd+="${IMAGE_NAME} "
cmd+="/bin/bash"

echo "$cmd" && \
eval "$cmd"

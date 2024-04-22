#!/bin/bash
# This script runs a local ML Flow Server.

# Parse input parameters
PORT=5000
while getopts "p:" opt; do
  case $opt in
    p) PORT=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Create directories
MLRUNS_DIR="$(pwd)/mlflow/mlruns"
ARTIFACTS_DIR="$(pwd)/mlflow/artifacts"
mkdir -p "$MLRUNS_DIR" && mkdir -p "$ARTIFACTS_DIR" &&

# Build command and run
cmd="docker run "
cmd+="-u root "
cmd+="--mount type=bind,source=$MLRUNS_DIR,target=/mlflow/mlruns "
cmd+="--mount type=bind,source=$ARTIFACTS_DIR,target=/mlflow/artifacts "
cmd+="-p $PORT:$PORT "
cmd+="--ulimit nproc=65535:65535 "
cmd+="-e OPENBLAS_NUM_THREADS=1 "
cmd+="-e GOTO_NUM_THREADS=1 "
cmd+="-e OMP_NUM_THREADS=1 "
cmd+="--rm "
cmd+="mlflow-server"

echo "$cmd" && \
eval "$cmd"

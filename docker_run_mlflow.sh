#!/bin/bash
# This script runs a local ML Flow Server.

PORT=5000

# Parse input parameters
while getopts "p:" opt; do
  case $opt in
    p) PORT=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# Build command and run
cmd="docker run "
cmd+="--mount type=bind,source=$(pwd)/mldlow/mlruns,target=/mlflow/mlruns "
cmd+="--mount type=bind,source=$(pwd)/mldlow/artifacts,target=/mlflow/artifacts "
cmd+="-p $PORT:5000 "
cmd+="--ulimit nproc=65535:65535 "
cmd+="-e OPENBLAS_NUM_THREADS=1 "
cmd+="-e GOTO_NUM_THREADS=1 "
cmd+="-e OMP_NUM_THREADS=1 "
cmd+="mlflow-server"

echo "$cmd" && \
eval "$cmd"

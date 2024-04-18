#!/bin/bash
# This script runs local ML Flow Server.

# Build command and run
cmd="docker run "
cmd+="--mount type=bind,source=$(pwd)/mlruns,target=/mlflow/mlruns "
cmd+="--mount type=bind,source=$(pwd)/artifacts,target=/mlflow/artifacts "
cmd+="-p 5000:5000 "
cmd+="--ulimit nproc=65535:65535 "
cmd+="-e OPENBLAS_NUM_THREADS=1 "
cmd+="-e GOTO_NUM_THREADS=1 "
cmd+="-e OMP_NUM_THREADS=1 "
cmd+="mlflow-server"

echo "$cmd" && \
eval "$cmd"

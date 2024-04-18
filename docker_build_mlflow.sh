#!/bin/bash
# This script builds local ML Flow Server docker image.

# Build command and run
cmd="docker build "
cmd+="-t mlflow-server "
cmd+="-f ./docker/local_ml_flow_server.Dockerfile"
cmd+=". "

echo "$cmd" && \
eval "$cmd"

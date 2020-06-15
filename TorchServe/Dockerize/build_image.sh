#!/bin/bash

MACHINE=cpu
BRANCH_NAME="master"
DOCKER_TAG="pytorch/torchserve:rz"

# rm -rf serve
# git clone https://github.com/pytorch/serve.git
# cd serve
# git checkout $BRANCH_NAME
# cd ..

# DOCKER_BUILDKIT=1 docker build --file Dockerfile_dev.$MACHINE -t $DOCKER_TAG .

# Rz :: Download and unpack serve from github https://github.com/pytorch/serve.git
DOCKER_BUILDKIT=1 docker build -t $DOCKER_TAG .
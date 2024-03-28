#!/bin/bash

IMAGE=opera/dswx-ni
tag=interface_0.1
echo "IMAGE is $IMAGE:$tag"

# fail on any non-zero exit codes
set -ex

python3 setup_ni.py sdist

# build image
docker build --rm --force-rm --network=host -t ${IMAGE}:$tag -f docker/Dockerfile .

# create image tar
docker save opera/dswx-ni > docker/dockerimg_dswx_ni_$tag.tar

# remove image
docker image rm opera/dswx-ni:$tag    

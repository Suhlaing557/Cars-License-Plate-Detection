#!/bin/bash

# Variables
IMAGE_NAME="cars_license_detection"
TAG="latest"
CONTAINER_NAME="car_license_detection"

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

# Run the Docker container
docker run -t -d --name ${CONTAINER_NAME} --rm --gpus all -v ./:/app/ ${IMAGE_NAME}:{TAG}

# Execute a bash shell in the running container
docker exec -it ${CONTAINER_NAME} /bin/bash

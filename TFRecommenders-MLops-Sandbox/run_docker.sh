#!/usr/bin/env bash

# Build image
#change tag for new container registery, gcr.io/bob
docker build -t ling-mlops-sandbox .

# List docker images
docker image ls

# Run flask app
docker run -p 127.0.0.1:8080:8080 ling-mlops-sandbox
#!/bin/bash

docker system prune
docker image rm marcondol/tensorflow-mlflow-opencv
docker build -t marcondol/tensorflow-mlflow-opencv .
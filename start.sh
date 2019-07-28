#!/usr/bin/env sh

cd docker
docker-compose run -w /home/tf/pytracer --rm pytracer

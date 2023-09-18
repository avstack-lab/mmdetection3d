#!/usr/bin/env bash

set -x

SKIPS=${1:?"missing arg 1 for SKIPS"}

python -m cProfile -o convert.prof convert_any_avstack_labels.py \
    --dataset carla \
    --subfolder ego-lidar/skip_"$SKIPS"/ \
    --data_dir /data/spencer/CARLA/ego-lidar \
    --n_skips "$SKIPS"
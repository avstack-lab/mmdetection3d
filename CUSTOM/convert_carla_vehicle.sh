#!/usr/bin/env bash

python -m cProfile -o convert.prof convert_any_avstack_labels.py \
    --dataset carla \
    --subfolder ego-lidar \
    --data_dir /data/spencer/CARLA/ego-lidar \
    --n_skips 0
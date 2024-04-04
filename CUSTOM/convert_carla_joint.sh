#!/usr/bin/env bash

set -x

SKIPS=${1:?"missing arg 1 for SKIPS"}

python -m cProfile -o convert.prof convert_any_avstack_labels.py \
    --dataset carla-joint \
    --subfolder joint/skip_"$SKIPS" \
    --data_dir /data/spencer/CARLA/multi-agent-v1 \
    --n_skips "$SKIPS"
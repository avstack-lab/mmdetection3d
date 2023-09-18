#!/usr/bin/env bash

set -x

MODEL=${1:?"missing arg 1 for MODEL"}


if [ $MODEL = "pointpillars" ]; then
    config="configs/carla/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-infrastructure.py"
elif [ $MODEL = "pointpillars-sec" ]; then
    config="configs/carla/pointpillars_hv_secfpn_sbn-all_8xb4-2x_carla-3d-infrastructure.py"
else
    echo "Incompatible model passed!" 1>&2
    exit 64
fi


CUDA_VISIBLE_DEVICES="0,1"
CONFIG="$config"
GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_carla_vehicle.py',
    '../_base_/datasets/carla-3d-vehicle.py', '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py'
]


# Optimizer
lr = 0.001  # 0.001 is for a batch size of 32 (?)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    # max_norm=10 is better for SECOND
    clip_grad=dict(max_norm=35, norm_type=2))
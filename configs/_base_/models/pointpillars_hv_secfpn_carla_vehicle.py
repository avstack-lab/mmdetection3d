_base_ = './pointpillars_hv_secfpn_waymo.py'

# model settings (based on nuScenes model settings)
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
pc_range = [-100, -100, -5, 100, 100, 5]
anchor_range = [-100, -100, -1.8, 100, 100, -1.8]
num_classes = 4

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(
            max_num_points=64,
            point_cloud_range=pc_range,
            max_voxels=(60000, 60000))),
    pts_voxel_encoder=dict(
        point_cloud_range=pc_range),
    pts_middle_encoder=dict(output_shape=[800, 800]),
    pts_bbox_head=dict(
        num_classes=num_classes,
        anchor_generator=dict(
            ranges=[anchor_range]),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9)),
    # model training settings (based on nuScenes model settings)
    train_cfg=dict(pts=dict(code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])))

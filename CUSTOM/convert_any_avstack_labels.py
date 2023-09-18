import os
import argparse
from tqdm import tqdm
import logging
import cv2
import numpy as np
import mmengine
from multiprocessing import Pool
from functools import partial
import json

import avapi

from avstack.environment.objects import Occlusion, ObjectStateDecoder
from avstack.geometry import GlobalOrigin3D, PointMatrix3D
from avstack.sensors import LidarData
from avstack import calibration
from avstack import maskfilters


def convert_avstack_to_annotation(SM, scene_splits, n_skips: int=1,
                                lidars: list=['LIDAR_TOP'], max_sweeps=10,
                                with_multi=False, n_max_proc=5, lidar_filter=None):
    """
    Converts avstack LiDAR data into annotation format
    
    INPUTS:
    SM -- scene manager
    """
    obj_accept = SM.nominal_whitelist_types
    obj_id_map = {o:i for i, o in enumerate(obj_accept)}

    # -- loop over scenes in this split
    n_problems = 0
    n_ignored = 0
    n_valid = 0
    idx_object = 0
    data_list = []
    for scene in tqdm(scene_splits):
        try:
            SD = SM.get_scene_dataset_by_name(scene)
        except IndexError as e:
            logging.exception(e)
            print(f'Could not process scene {scene}...continuing')
            continue

        # -- get names of sensors
        if lidars is None:
            lidars = [sens for sens in list(SD.sensors.keys()) if
                      (('lidar' in sens.lower()) or ('velo' in sens.lower())) and 
                      (sens != 'lidar') and (sens!='main_lidar')]
        if lidar_filter is not None:
            lidars = [li for li in lidars if lidar_filter.lower() in li.lower()]

        # -- loop over sensors
        for lid in lidars:
            # -- prep filepaths --> do it the long way to save for multiprocessing memory
            try:
                frames = SD.get_frames(sensor=lid)
            except KeyError:
                continue
            part_func = partial(process_frame, lid, obj_id_map)
            frames_all = [frames[idx] for idx in range(5, len(frames)-5, n_skips+1)]
            timestamps_all = [SD.get_timestamp(frame) for frame in frames_all]
            timestamp = None
            ego_filepaths = [
                SD.get_object_file(frame, timestamp, is_ego=True, is_global=True) for frame in frames_all
            ]
            objs_filepaths = [
                SD.get_object_file(frame, timestamp, sensor=lid, is_ego=False, is_global=False) for frame in frames_all
            ]
            calib_filepaths = [
                SD.get_sensor_file(frame, timestamp, lid, "calib") + ".txt" for frame in frames_all
            ]
            lidar_filepaths = [
                SD.get_sensor_file(frame, timestamp, lid, "data") + ".bin" for frame in frames_all
            ]

            # -- HACK: update filepaths in the case of infrastructure sensing
            if 'infra' in lid.lower():
                for i, frame in enumerate(frames_all):
                    lid_2 = lid + "_GROUNDED"
                    calib_filepaths[i] = calib_filepaths[i].replace('sensor_data', 'sensor_data_grounded').replace(lid, lid_2)
                    lidar_filepaths[i] = lidar_filepaths[i].replace('sensor_data', 'sensor_data_grounded').replace(lid, lid_2)

            # -- processing
            data_info_all = []
            instances_all = []
            if with_multi:
                chunksize = 1
                nproc = max(1, min(n_max_proc, int(len(frames_all)/chunksize)))
                zip_in = zip(frames_all, timestamps_all, ego_filepaths, objs_filepaths, calib_filepaths, lidar_filepaths)
                with Pool(nproc) as p:
                    for data_info, instances in tqdm(p.istarmap(part_func, zip_in, chunksize=chunksize), total=len(frames_all)):
                        data_info_all.append(data_info)
                        instances_all.append(instances)
            else:
                for frame, timestamp, ego_f, obj_f, calib_f, lidar_f in tqdm(zip(frames_all, timestamps_all, ego_filepaths, objs_filepaths, calib_filepaths, lidar_filepaths), total=len(frames_all)):
                    data_info, instances = part_func(frame, timestamp, ego_f, obj_f, calib_f, lidar_f)
                    data_info_all.append(data_info)
                    instances_all.append(instances)

            # -- package
            for frame, data_info, instances in zip(frames_all, data_info_all, instances_all):
                if data_info is None or instances is None:
                    n_problems += 1
                else:
                    data_info['scene'] = scene
                    data_info['frame'] = frame
                    data_list.append({'lidar_points':data_info, 'instances':instances, 'token':n_valid})
                    n_valid += 1

        # -- add idx object later
        pass

    # finish -- save and print outcomes
    print(f'{n_valid} valid point clouds; {n_problems} problems; {n_ignored} ignored with this set')
    return data_list, obj_id_map


def load_ego_from_file(filepath):
    with open(filepath, "r") as f:
        ego = json.load(f, cls=ObjectStateDecoder)
    return ego


def load_objects_from_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    objs = []
    for line in lines:
        line = line.rstrip()
        objs.append(json.loads(line, cls=ObjectStateDecoder))
    return np.asarray(objs)


def load_calibration_from_file(filepath):
    with open(filepath, "r") as f:
        calib = json.load(f, cls=calibration.CalibrationDecoder)
    return calib


def load_lidar_from_file(filepath, frame, ts, calib, sensor_ID):
    data = np.fromfile(filepath, dtype=np.float32).reshape(
        (-1, 4)
    )
    data = PointMatrix3D(data, calib)
    pc = LidarData(ts, frame, data, calib, sensor_ID)
    return pc


def process_frame(lid, obj_id_map, frame, timestamp, ego_filepath, obj_filepath, calib_filepath, lidar_filepath):
    # -- load information
    # ego = load_ego_from_file(ego_filepath)
    # ego_ref = ego.as_reference().differential(GlobalOrigin3D)
    objs = load_objects_from_file(obj_filepath)
    calib = load_calibration_from_file(calib_filepath)
    # pc = load_lidar_from_file(lidar_filepath, frame, timestamp, calib, sensor_ID=0)

    # -- data info
    data_info = dict()
    data_info['lidar_path'] = lidar_filepath
    data_info['num_pts_feats'] = None

    # -- annotation information
    instances = []
    for obj in objs:
        if obj.occlusion in [Occlusion.COMPLETE]:
            continue
        bbox_3d = obj.box

        # HACK: update objects in the case of infrastructure sensing
        if 'infra' in lid.lower():
            bbox_3d.change_reference(calib.reference)

        # -- store annotations
        ann_info = dict()  # needs to be inside the loop for copying purposes
        ann_info['distance'] = bbox_3d.position.norm()
        ann_info['volume'] = np.product(bbox_3d.hwl)
        ann_info['bbox_label_3d'] = obj_id_map[obj.obj_type]
        ann_info['bbox_3d'] = [*bbox_3d.position.x, *reversed(bbox_3d.hwl), *[bbox_3d.yaw, 0, 0]]
        # ann_info['bbox_3d'][2] = 0  # HACK: consider the bottom of the box.
        ann_info['num_lidar_pts'] = 400  # HACK: make this wayyy faster by ignoring sum(maskfilters.filter_points_in_box(pc, bbox_3d.corners))
        ann_info['num_radar_pts'] = 0
        ann_info['velocity'] = obj.velocity.x[:2]
        ann_info['token'] = -1

        # -- merge infos
        instances.append(ann_info)

    return data_info, instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wrap avstack data to nuscenes format for training')
    parser.add_argument('--dataset', choices=['carla', 'carla-infrastructure', 'carla-joint', 'kitti', 'nuscenes'], help='Choice of dataset')
    parser.add_argument('--subfolder', type=str, help='Save subfolder name')
    parser.add_argument('--data_dir', type=str, help='Path to main dataset storage location')
    parser.add_argument('--n_skips', default=0, type=int, help='Number of skips between frames of a sequence')
    args = parser.parse_args()

    # -- create scene manager and get scene splits
    dataset = args.dataset
    lidar_filter = None
    if args.dataset == 'carla':
        SM = avapi.carla.CarlaScenesManager(args.data_dir)
        lidars = None
        splits_scenes = avapi.carla.get_splits_scenes(args.data_dir)
    elif args.dataset == 'carla-infrastructure':
        dataset = 'carla'
        lidar_filter = 'infra'
        SM = avapi.carla.CarlaScenesManager(args.data_dir)
        lidars = None
        splits_scenes = avapi.carla.get_splits_scenes(args.data_dir)
    elif args.dataset == 'carla-joint':
        dataset = 'carla'
        lidars = None
        SM = avapi.carla.CarlaScenesManager(args.data_dir)
        splits_scenes = avapi.carla.get_splits_scenes(args.data_dir)
    elif args.dataset == 'kitti':
        lidars = ['velodyne']
        splits_scenes = avapi.kitti.splits_scenes
        raise
    elif args.dataset == 'nuscenes':
        SM = avapi.nuscenes.nuScenesManager(args.data_dir, split="v1.0-trainval")
        lidars = ['LIDAR_TOP']
        splits_scenes = avapi.nuscenes.splits_scenes
    else:
        raise NotImplementedError(args.dataset)

    # -- run main call
    for split in ['train', 'val']:
        print(f'Converting {split}...')
        out_file = f'../data/{dataset}/{args.subfolder}/{split}_annotation_{dataset}_in_nuscenes.pkl'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        data_list, obj_id_map = convert_avstack_to_annotation(SM, splits_scenes[split], n_skips=args.n_skips, lidars=lidars, lidar_filter=lidar_filter)
        metainfo = dict(
            categories=obj_id_map,
            dataset=args.dataset,
            version=split,
            info_version="1.1"
        )
        data = dict(data_list=data_list, metainfo=metainfo)
        mmengine.dump(data, out_file)
        print(f'done')
import copy
import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2
import pickle

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils_2 import show_image
import trimesh
import matplotlib

import mesh_utils

def get_rgb_colors():
    rgb_colors = []
    # each color is a tuple of (name, (r,g,b))
    for name, hex in matplotlib.colors.cnames.items():
        rgb_colors.append((name, matplotlib.colors.to_rgb(hex)))

    rgb_colors = sorted(rgb_colors, key=lambda x: x[0])

    priority_colors = [('red', (1.0, 0.0, 0.0)),  ('green', (0.0, 1.0, 0.0)), ('blue', (0.0, 0.0, 1.0)),  ('orange', (1.0, 0.6470588235294118, 0.0)),  ('purple', (0.5019607843137255, 0.0, 0.5019607843137255)),  ('magenta', (1.0, 0.0, 1.0)),]
    rgb_colors = priority_colors + rgb_colors

    return rgb_colors

def inference(global_config, checkpoint_dir, input_paths, K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    os.makedirs('results', exist_ok=True)

    # Process example test scenes
    for p in glob.glob(input_paths):
        print('Loading ', p)

        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
        
        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, z_range=z_range)

        print("pc_full", pc_full.shape)
        print("pc_colors", pc_colors.shape)
        print("pc_segments", len(pc_segments))
        trimesh.PointCloud(pc_full, colors=pc_colors).show()
        pcs_vis = []
        rgb_colors = get_rgb_colors()
        print(len(rgb_colors))
        print(np.array(list(rgb_colors[0][1]) + [1], dtype=np.float))
        for seg_id in pc_segments:
            pcs_vis.append(trimesh.PointCloud(pc_segments[seg_id], colors=list(rgb_colors[int(seg_id)][1]) + [1]))
        trimesh.Scene(pcs_vis).show()

        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

        # Save results
        np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                  pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

        # Visualize results          
        # show_image(rgb, segmap)
        # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

        gripper = mesh_utils.create_gripper('panda')
        gripper_mesh = gripper.hand
        for i, k in enumerate(pred_grasps_cam):
            for j in range(min(len(pred_grasps_cam[k]), 10)):
                tfm = pred_grasps_cam[k][j]
                print(pred_grasps_cam[k][j])
                gm = copy.deepcopy(gripper_mesh)
                gm.apply_transform(tfm)
                trimesh.Scene(pcs_vis + [gm]).show()

        input("next scene?")
        
    if not glob.glob(input_paths):
        print('No files found: ', input_paths)


def inference_on_pc(global_config, checkpoint_dir, input_filename, output_filename, K=None, local_regions=True, skip_border_objects=False,
              filter_grasps=True, segmap_id=None, z_range=[0.2, 1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data

    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments.
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """

    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')

    os.makedirs('results', exist_ok=True)

    # Process example test scenes
    with open(input_filename, "rb") as fh:
        data_dict = pickle.load(fh, encoding="bytes")
    # data_dict = np.load(input_paths, allow_pickle=True, encoding="bytes")
    print(data_dict.keys())
    pc_full = data_dict[bytes("pc_full", 'utf-8')]
    pc_colors = data_dict[bytes("pc_colors", 'utf-8')]
    pc_segments = data_dict[bytes("pc_segments", 'utf-8')]

    print("pc_full", pc_full.shape)
    print("pc_colors", pc_colors.shape)
    print("pc_segments", len(pc_segments))
    trimesh.PointCloud(pc_full, colors=pc_colors).show()
    pcs_vis = []
    rgb_colors = get_rgb_colors()
    print(len(rgb_colors))
    print(np.array(list(rgb_colors[0][1]) + [1], dtype=np.float))
    for seg_id in pc_segments:
        pcs_vis.append(trimesh.PointCloud(pc_segments[seg_id], colors=list(rgb_colors[int(seg_id)][1]) + [1]))
    trimesh.Scene(pcs_vis).show()

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full,
                                                                                   pc_segments=pc_segments,
                                                                                   local_regions=local_regions,
                                                                                   filter_grasps=filter_grasps,
                                                                                   forward_passes=forward_passes)

    # Save results
    with open(output_filename, "wb") as fh:
        pickle.dump({"pred_grasps_cam": pred_grasps_cam, "scores": scores, "contact_pts": contact_pts}, fh, protocol=2)
    # np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png', 'npz').replace('npy', 'npz'))),
    #          pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

    # Visualize results
    # show_image(rgb, segmap)
    # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)

    # gripper = mesh_utils.create_gripper('panda')
    # gripper_mesh = gripper.hand
    # gms = []
    # grasps_vis = []
    # for i, k in enumerate(pred_grasps_cam):
    #     for j in range(min(len(pred_grasps_cam[k]), 10)):
    #         tfm = pred_grasps_cam[k][j]
    #         print(pred_grasps_cam[k][j])
    #         gm = copy.deepcopy(gripper_mesh)
    #         gm.apply_transform(tfm)
    #         gms.append(gm)
    #         grasp_vis = trimesh.creation.axis(transform=tfm, origin_size=0.001, axis_radius=0.001, axis_length=0.05)
    #         grasps_vis.append(grasp_vis)
    # trimesh.Scene(pcs_vis + gms + grasps_vis).show()
    # trimesh.Scene(pcs_vis + grasps_vis).show()

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    parser.add_argument('--temp_pc_filename', default='', help='')
    parser.add_argument('--temp_grasp_filename', default='', help='')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

    # inference(global_config, FLAGS.ckpt_dir, FLAGS.np_path if not FLAGS.png_path else FLAGS.png_path, z_range=eval(str(FLAGS.z_range)),
    #             K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id,
    #             forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)

    inference_on_pc(global_config, FLAGS.ckpt_dir, FLAGS.temp_pc_filename, FLAGS.temp_grasp_filename, z_range=eval(str(FLAGS.z_range)),
                K=FLAGS.K, local_regions=FLAGS.local_regions, filter_grasps=FLAGS.filter_grasps, segmap_id=FLAGS.segmap_id,
                forward_passes=FLAGS.forward_passes, skip_border_objects=FLAGS.skip_border_objects)


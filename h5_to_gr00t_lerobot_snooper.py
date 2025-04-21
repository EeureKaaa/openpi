#!/usr/bin/env python3

"""
Script to convert RLDS dataset to GR00T-compatible LeRobot format.

This script handles:
1. Creating the proper directory structure
2. Generating videos from image frames
3. Creating parquet files with the required columns
4. Generating all necessary metadata files (modality.json, tasks.jsonl, episodes.jsonl, info.json)

Usage:
python h5_to_gr00t_lerobot.py --data_dir /path/to/h5/data --output_dir /path/to/output/directory

Required columns in GR00T LeRobot format:
- observation.state
- action
- timestamp
- annotation.human.action.task_description
- task_index
- annotation.human.validity
- episode_index
- index
- next.reward
- next.done
"""
import pysnooper
import os
import cv2
import numpy as np
import pandas as pd
import json
import tensorflow_datasets as tfds
import argparse
from tqdm import tqdm
import shutil
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import io
from PIL import Image
import time
import h5py
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert h5 dataset to GR00T-compatible LeRobot format')
    parser.add_argument('--data_dir', type=str, default='/home/wangxianhao/data/project/reasoning/coin_filtered',
                        help='Directory containing environment directories with h5 files')
    parser.add_argument('--output_dir', type=str, default='/home/wangxianhao/data/project/reasoning/openpi/gr00t_dataset',
                        help='Output directory for GR00T LeRobot dataset')
    parser.add_argument('--dataset_name', type=str,
                        default='primitive_dataset', help='Name of the dataset')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for generated videos')
    parser.add_argument('--resize_dim', type=int, default=256,
                        help='Resize dimension for images')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Number of episodes per chunk')
    parser.add_argument('--state_dim', type=int, default=8,
                        help='Dimension of the state vector (qpos)')
    parser.add_argument('--action_dim', type=int, default=7,
                        help='Dimension of the action vector')
    parser.add_argument('--generate_videos', action='store_true',
                        help='Generate videos from frames')
    parser.add_argument('--inst_path', type=str, default='/home/wangxianhao/data/project/reasoning/openpi/env_ins.json',
                        help='Path to the environment instructions JSON file')
    return parser.parse_args()


def create_directory_structure(output_dir):
    """Create the necessary directory structure for GR00T LeRobot format."""
    os.makedirs(os.path.join(output_dir, 'meta'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data', 'chunk-000'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos', 'chunk-000',
                'observation.images.base_view'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos', 'chunk-000',
                'observation.images.wrist_view'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos', 'chunk-000',
                'observation.images.left_view'), exist_ok=True)

    print(f"Created directory structure in {output_dir}")


def create_video_from_frames(frames, output_path, fps=10):
    """Create an MP4 video from a list of frames."""
    if not frames:
        print(f"Warning: No frames to create video at {output_path}")
        return False

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Check if image is grayscale or color
    is_color = len(frames[0].shape) == 3 and frames[0].shape[2] == 3
    video = cv2.VideoWriter(output_path, fourcc, fps,
                            (width, height), isColor=is_color)

    # Write frames to video
    for frame in frames:
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Handle color conversion if needed
        if is_color:
            # OpenCV uses BGR, but images might be RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        video.write(frame)

    # Release video writer
    video.release()
    return True


def create_modality_json(output_dir, args):
    """Create the modality.json file for GR00T LeRobot format."""
    modality_config = {
        "state": {
            "joint_state": {
                "start": 0,
                "end": args.state_dim
            }
        },
        "action": {
            "ee_action": {
                "start": 0,
                "end": args.action_dim
            }
        },
        "video": {
            "base_view": {
                "original_key": "observation.images.base_view"
            },
            "wrist_view": {
                "original_key": "observation.images.wrist_view"
            },
            "left_view": {
                "original_key": "observation.images.left_view"
            },
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {}
        }
    }

    with open(os.path.join(output_dir, 'meta', 'modality.json'), 'w') as f:
        json.dump(modality_config, f, indent=4)

    print(f"Created modality.json in {output_dir}/meta")


def create_info_json(output_dir, dataset_name, args, total_episodes, total_frames, total_tasks):
    """Create the info.json file for GR00T LeRobot format with the proper features field."""
    # Basic dataset information
    info = {
        "name": dataset_name,
        "description": f"GR00T-compatible LeRobot dataset converted from RLDS {dataset_name}",
        "version": "1.0.0",
        "license": "apache-2.0",
        "author": "Converted using rlds_to_gr00t_lerobot.py",
        "homepage": "https://github.com/NVIDIA/Isaac-GR00T",
        "repository": "",
        "citation": "",
        "tags": ["robot", "manipulation", "gr00t"],

        # Required GR00T fields
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * 3,  # Base, wrist and left views
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
        "splits": {
            "train": "0:100"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",

        # Features field required by GR00T
        "features": {
            "observation.images.base_view": {
                "dtype": "video",
                "shape": [args.resize_dim, args.resize_dim, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": args.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.wrist_view": {
                "dtype": "video",
                "shape": [args.resize_dim, args.resize_dim, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": args.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.left_view": {
                "dtype": "video",
                "shape": [args.resize_dim, args.resize_dim, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": args.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [args.state_dim],
                "names": [f"state_{i}" for i in range(args.state_dim)]
            },
            "action": {
                "dtype": "float64",
                "shape": [args.action_dim],
                "names": [f"action_{i}" for i in range(args.action_dim)]
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [1]
            },
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1]
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "annotation.human.validity": {
                "dtype": "int64",
                "shape": [1]
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1]
            },
            "index": {
                "dtype": "int64",
                "shape": [1]
            },
            "next.reward": {
                "dtype": "float64",
                "shape": [1]
            },
            "next.done": {
                "dtype": "float64",
                "shape": [1]
            }
        }
    }

    with open(os.path.join(output_dir, 'meta', 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)

    print(f"Created info.json with features field in {output_dir}/meta")


def process_episode(h5_dataset, episode_idx, output_dir, args, env_name=None, task_instruction=None):
    """Process a single episode from h5 and convert to GR00T LeRobot format."""
    # Get reference to trajectory group
    traj_group = h5_dataset['traj_0']

    # Extract dimensions
    actions = traj_group['actions']
    num_actions = actions.shape[0]
    print(f"Number of actions: {num_actions}")

    # Skip episodes with too few actions
    if num_actions < 8:
        print(
            f"Warning: Episode {episode_idx} has only {num_actions} actions (less than 8), skipping")
        return None

    # Create lists to store frames for video generation
    base_frames = []  # base camera
    wrist_frames = []  # wrist camera
    left_frames = []  # left camera

    # Create lists to store data for parquet file
    observation_states = []
    actions_list = []
    timestamps = []
    task_indices = []
    next_rewards = []
    next_dones = []

    # Process each step in the episode (using action length as our step count)
    for i in range(num_actions):
        # Extract camera data
        base_img = cv2.resize(traj_group['obs']['sensor_data']['base_camera']['rgb'][i],
                              (args.resize_dim, args.resize_dim),
                              interpolation=cv2.INTER_LANCZOS4)
        wrist_img = cv2.resize(traj_group['obs']['sensor_data']['hand_camera']['rgb'][i],
                               (args.resize_dim, args.resize_dim),
                               interpolation=cv2.INTER_LANCZOS4)
        left_img = cv2.resize(traj_group['obs']['sensor_data']['left_camera']['rgb'][i],
                              (args.resize_dim, args.resize_dim),
                              interpolation=cv2.INTER_LANCZOS4)

        # Frames
        base_frames.append(base_img)
        wrist_frames.append(wrist_img)
        left_frames.append(left_img)

        # Extract state and action
        state = traj_group['obs']['agent']['qpos'][i][:8]
        action = actions[i]

        # Add data for parquet file
        observation_states.append(state)
        actions_list.append(action)
        # Calculate timestamp based on frame index and fps
        timestamps.append(float(i) / args.fps)
        # Will be updated later with the correct task index
        task_indices.append(0)

        # Extract reward and done flag for next step
        next_rewards.append(float(traj_group['rewards'][i]))
        next_dones.append(
            bool(traj_group['terminated'][i] or traj_group['truncated'][i]))

    # Generate videos if needed
    base_video_success = True
    wrist_video_success = True
    left_video_success = True

    # Generate videos
    base_video_path = os.path.join(output_dir, 'videos', 'chunk-000',
                                   'observation.images.base_view', f'episode_{episode_idx:06d}.mp4')
    wrist_video_path = os.path.join(output_dir, 'videos', 'chunk-000',
                                    'observation.images.wrist_view', f'episode_{episode_idx:06d}.mp4')
    left_video_path = os.path.join(output_dir, 'videos', 'chunk-000',
                                   'observation.images.left_view', f'episode_{episode_idx:06d}.mp4')

    base_video_success = create_video_from_frames(
        base_frames, base_video_path, args.fps)
    wrist_video_success = create_video_from_frames(
        wrist_frames, wrist_video_path, args.fps)
    left_video_success = create_video_from_frames(
        left_frames, left_video_path, args.fps)

    if not (base_video_success and wrist_video_success and left_video_success):
        print(
            f"Warning: Failed to create videos for episode {episode_idx}, skipping")
        return None

    # Create DataFrame for parquet file
    df_data = {
        "observation.state": observation_states,
        "action": actions_list,
        "timestamp": timestamps,
        "task_index": task_indices,
        "episode_index": [episode_idx] * num_actions,
        "index": list(range(num_actions)),
        "next.reward": next_rewards,
        "next.done": next_dones,
        # We'll add annotation columns later after we have the task indices
    }

    # Return episode data
    return {
        "episode_idx": episode_idx,
        "task_description": task_instruction,
        "length": num_actions,
        "df_data": df_data
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir) / args.dataset_name

    # Create directory structure
    create_directory_structure(output_dir)

    # Load environment instructions
    env_instructions = {}
    if os.path.exists(args.inst_path):
        try:
            with open(args.inst_path, 'r') as f:
                env_instructions = json.load(f)
            print(
                f"Loaded {len(env_instructions)} environment instructions from {args.inst_path}")
        except Exception as e:
            print(f"Error loading environment instructions: {e}")
            env_instructions = {}

    # Get all environment directories
    env_dirs = [d for d in os.listdir(args.data_dir)
                if os.path.isdir(os.path.join(args.data_dir, d))
                and d.startswith('Tabletop-')]

    print(f"Found {len(env_dirs)} environment directories in {args.data_dir}")
    input("Press Enter to continue...")

    # Process all environments
    episodes_data = []
    tasks = {}
    task_index_counter = 0
    episode_idx = 0

    # Process each environment directory
    for env_name in env_dirs:
        env_dir = os.path.join(args.data_dir, env_name)
        print(f"\nProcessing environment: {env_name}")

        # Get instruction for this environment
        task_instruction = env_instructions.get(
            env_name, f"Task for {env_name}")

        # Check if this task already exists
        if task_instruction not in tasks:
            tasks[task_instruction] = task_index_counter
            task_index_counter += 1

        task_idx = tasks[task_instruction]
        print(f"Task: {task_instruction} (Index: {task_idx})")

        # Find all h5 files in this environment directory
        h5_files = glob.glob(os.path.join(
            env_dir, '**', '*.h5'), recursive=True)
        print(f"Found {len(h5_files)} h5 files in {env_dir}")

        # Process each h5 file in this environment
        for h5_file in h5_files:
            print(f"Processing {h5_file}...")
            start_time = time.time()

            try:
                with h5py.File(h5_file, 'r') as h5_dataset:
                    episode_data = process_episode(h5_dataset, episode_idx, output_dir, args,
                                                   env_name=env_name, task_instruction=task_instruction)

                    if episode_data:
                        # Use the task index we already determined for this environment
                        for i in range(len(episode_data["df_data"]["task_index"])):
                            episode_data["df_data"]["task_index"][i] = task_idx

                        # Add annotation columns
                        episode_data["df_data"]["annotation.human.action.task_description"] = [
                            task_idx] * len(episode_data["df_data"]["task_index"])
                        episode_data["df_data"]["annotation.human.validity"] = [
                            1] * len(episode_data["df_data"]["task_index"])  # Assuming all data is valid

                        episodes_data.append(episode_data)
                        episode_idx += 1  # Increment episode index for next episode
            except Exception as e:
                print(f"Error processing {h5_file}: {e}")
                continue

            end_time = time.time()
            print(
                f"Finished processing {h5_file} in {end_time - start_time:.2f} seconds")

    # Create tasks.jsonl
    with open(os.path.join(output_dir, 'meta', 'tasks.jsonl'), 'w') as f:
        for task, task_idx in tasks.items():
            f.write(json.dumps({"task_index": task_idx, "task": task}) + '\n')

    print(f"Created tasks.jsonl with {len(tasks)} tasks")

    # Create episodes.jsonl
    with open(os.path.join(output_dir, 'meta', 'episodes.jsonl'), 'w') as f:
        for episode_data in episodes_data:
            episode_entry = {
                "episode_index": episode_data["episode_idx"],
                # Assuming all episodes are valid
                "tasks": [episode_data["task_description"], "valid"],
                "length": episode_data["length"]
            }
            f.write(json.dumps(episode_entry) + '\n')

    print(f"Created episodes.jsonl with {len(episodes_data)} episodes")

    # Create parquet files
    for episode_data in tqdm(episodes_data, desc="Creating parquet files"):
        df = pd.DataFrame(episode_data["df_data"])

        # Ensure columns are in the correct order for GR00T
        columns_order = [
            "observation.state", "action", "timestamp",
            "annotation.human.action.task_description", "task_index",
            "annotation.human.validity", "episode_index", "index",
            "next.reward", "next.done"
        ]

        # Reorder columns and save parquet file
        df = df[columns_order]

        # Save parquet file
        parquet_path = os.path.join(
            output_dir, 'data', 'chunk-000', f'episode_{episode_data["episode_idx"]:06d}.parquet')
        df.to_parquet(parquet_path, index=False)

    # Create modality.json using the state and action dimensions from args
    create_modality_json(output_dir, args)

    # Calculate total frames and tasks for info.json
    total_episodes = len(episodes_data)
    total_frames = sum(episode['length'] for episode in episodes_data)
    total_tasks = len(tasks)

    # Create info.json with the proper features field
    create_info_json(output_dir, args.dataset_name, args,
                     total_episodes, total_frames, total_tasks)

    print(
        f"Conversion complete! GR00T-compatible LeRobot dataset created in {output_dir}")


if __name__ == "__main__":
    import glob
    import sys
    print("Starting conversion to GR00T LeRobot format...")
    try:
        main()
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

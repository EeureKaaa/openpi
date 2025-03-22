#!/usr/bin/env python3
"""
Script to convert RLDS dataset to GR00T-compatible LeRobot format.

This script handles:
1. Creating the proper directory structure
2. Generating videos from image frames
3. Creating parquet files with the required columns
4. Generating all necessary metadata files (modality.json, tasks.jsonl, episodes.jsonl, info.json)

Usage:
python rlds_to_gr00t_lerobot.py --data_dir /path/to/rlds/data --output_dir /path/to/output/directory

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

def parse_args():
    parser = argparse.ArgumentParser(description='Convert RLDS dataset to GR00T-compatible LeRobot format')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing RLDS dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for GR00T LeRobot dataset')
    parser.add_argument('--dataset_name', type=str, default='tabletop_dataset', help='Name of the RLDS dataset')
    parser.add_argument('--fps', type=int, default=10, help='FPS for generated videos')
    parser.add_argument('--resize_dim', type=int, default=256, help='Resize dimension for images')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Number of episodes per chunk')
    parser.add_argument('--state_dim', type=int, default=8, help='Dimension of the state vector')
    parser.add_argument('--action_dim', type=int, default=7, help='Dimension of the action vector')
    return parser.parse_args()

def create_directory_structure(output_dir):
    """Create the necessary directory structure for GR00T LeRobot format."""
    os.makedirs(os.path.join(output_dir, 'meta'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data', 'chunk-000'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos', 'chunk-000', 'observation.images.ego_view'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos', 'chunk-000', 'observation.images.wrist_view'), exist_ok=True)
    
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
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)
    
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
            "ego_view": {
                "original_key": "observation.images.ego_view"
            },
            "wrist_view": {
                "original_key": "observation.images.wrist_view"
            }
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
        "total_videos": total_episodes * 2,  # Ego and wrist views
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
            "observation.images.ego_view": {
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

def process_episode(episode, episode_idx, output_dir, args):
    """Process a single episode from RLDS and convert to GR00T LeRobot format."""
    # Extract steps from the episode
    steps = list(episode["steps"].as_numpy_iterator())
    
    # Skip empty episodes
    if not steps:
        print(f"Warning: Episode {episode_idx} is empty, skipping")
        return None
    
    # Extract task description
    task_description = steps[0]["language_instruction"].decode()
    
    # Create lists to store frames for video generation
    main_frames = []
    wrist_frames = []
    
    # Create lists to store data for parquet file
    observation_states = []
    actions = []
    timestamps = []
    task_indices = []
    next_rewards = []
    next_dones = []
    
    # Process each step in the episode
    for i, step in enumerate(steps):
        # Extract and resize images
        main_img = cv2.resize(step["observation"]["image"], 
                             (args.resize_dim, args.resize_dim), 
                             interpolation=cv2.INTER_LANCZOS4)
        wrist_img = cv2.resize(step["observation"]["wrist_image"], 
                              (args.resize_dim, args.resize_dim), 
                              interpolation=cv2.INTER_LANCZOS4)
        
        # Add frames for video generation
        main_frames.append(main_img)
        wrist_frames.append(wrist_img)
        
        # Extract state and action
        state = step["observation"]["state"]
        action = step["action"]
        
        # Add data for parquet file
        observation_states.append(state)
        actions.append(action)
        timestamps.append(float(i) / args.fps)  # Calculate timestamp based on frame index and fps
        task_indices.append(0)  # Will be updated later with the correct task index
        
        # Extract reward and done flag for next step
        next_rewards.append(float(step["reward"]))
        next_dones.append(bool(step["is_last"]))
    
    # Generate videos
    main_video_path = os.path.join(output_dir, 'videos', 'chunk-000', 'observation.images.ego_view', f'episode_{episode_idx:06d}.mp4')
    wrist_video_path = os.path.join(output_dir, 'videos', 'chunk-000', 'observation.images.wrist_view', f'episode_{episode_idx:06d}.mp4')
    
    main_video_success = create_video_from_frames(main_frames, main_video_path, args.fps)
    wrist_video_success = create_video_from_frames(wrist_frames, wrist_video_path, args.fps)
    
    if not (main_video_success and wrist_video_success):
        print(f"Warning: Failed to create videos for episode {episode_idx}, skipping")
        return None
    
    # Create DataFrame for parquet file
    df_data = {
        "observation.state": observation_states,
        "action": actions,
        "timestamp": timestamps,
        "task_index": task_indices,
        "episode_index": [episode_idx] * len(steps),
        "index": list(range(len(steps))),
        "next.reward": next_rewards,
        "next.done": next_dones,
        # We'll add annotation columns later after we have the task indices
    }
    
    # Return episode data
    return {
        "episode_idx": episode_idx,
        "task_description": task_description,
        "length": len(steps),
        "df_data": df_data
    }

def main():
    args = parse_args()

    output_dir = Path(args.output_dir) / args.dataset_name
    
    # Create directory structure
    create_directory_structure(output_dir)
    
    # Load RLDS dataset
    print(f"Loading RLDS dataset from {args.data_dir}/{args.dataset_name}")
    rlds_dataset = tfds.load(args.dataset_name, data_dir=args.data_dir, split="train")
    
    # Process episodes
    episodes_data = []
    tasks = {}
    task_index_counter = 0
    
    for episode_idx, episode in enumerate(tqdm(rlds_dataset, desc="Processing episodes")):
        episode_data = process_episode(episode, episode_idx, output_dir, args)
        
        if episode_data:
            # Check if task already exists, if not add it to tasks
            task_description = episode_data["task_description"]
            if task_description not in tasks:
                tasks[task_description] = task_index_counter
                task_index_counter += 1
            
            # Update task_index in DataFrame
            task_idx = tasks[task_description]
            for i in range(len(episode_data["df_data"]["task_index"])):
                episode_data["df_data"]["task_index"][i] = task_idx
            
            # Add annotation columns
            episode_data["df_data"]["annotation.human.action.task_description"] = [task_idx] * len(episode_data["df_data"]["task_index"])
            episode_data["df_data"]["annotation.human.validity"] = [1] * len(episode_data["df_data"]["task_index"])  # Assuming all data is valid
            
            episodes_data.append(episode_data)
    
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
                "tasks": [episode_data["task_description"], "valid"],  # Assuming all episodes are valid
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
        parquet_path = os.path.join(output_dir, 'data', 'chunk-000', f'episode_{episode_data["episode_idx"]:06d}.parquet')
        df.to_parquet(parquet_path, index=False)
    
    # Create modality.json using the state and action dimensions from args
    create_modality_json(output_dir, args)
    
    # Calculate total frames and tasks for info.json
    total_episodes = len(episodes_data)
    total_frames = sum(episode['length'] for episode in episodes_data)
    total_tasks = len(tasks)
    
    # Create info.json with the proper features field
    create_info_json(output_dir, args.dataset_name, args, total_episodes, total_frames, total_tasks)
    
    print(f"Conversion complete! GR00T-compatible LeRobot dataset created in {output_dir}")

if __name__ == "__main__":
    main()

from typing import Iterator, Tuple, Any

import copy
import os
import cv2
import glob
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from calvin_dataset.conversion_utils import MultiThreadedDatasetBuilder


FILE_PATH = '/nfs/kun2/datasets/calvin'


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(ids):
        episode = []
        for i, id in enumerate(range(ids[0], ids[1]+1)):
            try:
                data = np.load(os.path.join(FILE_PATH, f'episode_{id:07}.npz'))
            except:
                print(f"Failed to load {os.path.join(FILE_PATH, f'episode_{id:07}.npz')}")
                return None

            episode.append({
                'observation': {
                    'rgb_static': data['rgb_static'],
                    'rgb_gripper': data['rgb_gripper'],
                    'rgb_tactile': data['rgb_tactile'],
                    'depth_static': data['rgb_static'],
                    'depth_gripper': data['rgb_gripper'],
                    'depth_tactile': data['rgb_tactile'],
                    'robot_obs': np.asarray(data['robot_obs'], dtype=np.float32),
                    'scene_obs': np.asarray(data['scene_obs'], dtype=np.float32),
                    'natural_language_instruction': annotation,
                },
                'action': {
                    'actions': np.asarray(data['actions'], dtype=np.float32),
                    'rel_actions_world': np.asarray(data['rel_actions_world'], dtype=np.float32),
                },
                'discount': 1.0,
                'reward': float(i == (ids[1]-ids[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (ids[1]-ids[0] - 1),
                'is_terminal': i == (ids[1]-ids[0] - 1),
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return str(ids), sample

    # for smallish datasets, use single-thread parsing
    for ids, annotation in paths:
        yield _parse_example(ids, annotation)


class Calvin(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40              # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'rgb_static': tfds.features.Image(
                            shape=(200, 200, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='RGB camera observation.',
                        ),
                        'rgb_gripper': tfds.features.Image(
                            shape=(84, 84, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='RGB gripper camera observation.',
                        ),
                        'rgb_tactile': tfds.features.Image(
                            shape=(160, 120, 6),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='RGB tactile camera observation.',
                        ),
                        'depth_static': tfds.features.Tensor(
                            shape=(200, 200),
                            dtype=np.float32,
                            doc='Main camera depth observation.',
                        ),
                        'depth_gripper': tfds.features.Tensor(
                            shape=(84, 84),
                            dtype=np.float32,
                            doc='Wrist camera depth observation.',
                        ),
                        'depth_tactile': tfds.features.Tensor(
                            shape=(160, 120, 2),
                            dtype=np.float32,
                            doc='Wrist camera depth observation.',
                        ),
                        'robot_obs': tfds.features.Tensor(
                            shape=(15,),
                            dtype=np.float32,
                            doc='EE position (3), EE orientation in euler angles (3), '
                                'gripper width (1), joint positions (7), gripper action (1).',
                        ),
                        'scene_obs': tfds.features.Tensor(
                            shape=(24,),
                            dtype=np.float32,
                            doc='sliding door (1): joint state, '
                                'drawer (1): joint state, '
                                'button (1): joint state, '
                                'switch (1): joint state, '
                                'lightbulb (1): on=1, off=0, '
                                'green light (1): on=1, off=0, '
                                'red block (6): (x, y, z, euler_x, euler_y, euler_z), '
                                'blue block (6): (x, y, z, euler_x, euler_y, euler_z), '
                                'pink block (6): (x, y, z, euler_x, euler_y, euler_z), '
                        ),
                        'natural_language_instruction': tfds.features.Text(
                            doc='Language Instruction.'
                        ),
                    }),
                    'action': tfds.features.FeaturesDict({
                        'actions': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='absolute desired values for gripper pose '
                                '(first 3 dimensions are x, y, z in absolute world coordinates,'
                                'next 3 dimensions  are euler angles x,y,z), '
                                'last dimension is open_gripper (-1 is open close, 1 is open).',
                        ),
                        'rel_actions_world': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='relative actions for gripper pose in the robot base frame '
                                '(first 3 dimensions are x, y, z  in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 50'
                                'next 3 dimensions are euler angles x,y,z  in relative world coordinates normalized and clipped to (-1, 1) with scaling factor 20), '
                                'last dimension is open_gripper (-1 is close gripper, 1 is open).',
                        ),
                    }),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),

                }),
                'episode_metadata': tfds.features.FeaturesDict({
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        train_annotations = np.load(os.path.join(FILE_PATH, "training/ep_start_end_ids.npy"), allow_pickle=True)
        val_annotations = np.load(os.path.join(FILE_PATH, "validation/ep_start_end_ids.npy"), allow_pickle=True)

        return {
            'train': train_annotations.tolist(),
            'val': val_annotations.tolist(),
        }

# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import os
import glob
import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class NsvfDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: NSVF)
    """target class to instantiate"""
    data: Path = Path("data/NSVF/Synthetic_NSVF/Bike")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class NSVF(DataParser):

    config: NsvfDataParserConfig

    def __init__(self, config: NsvfDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.root_dir = str(self.data)
        self.downsample = 1
        self.fx, self.fy, self.cx, self.cy = self.read_intrinsics()
        self.shift, self.scale = self.read_scale_shift()
        
    def read_scale_shift(self):
        xyz_min, xyz_max = np.loadtxt(os.path.join(self.root_dir, 'bbox.txt'))[:6].reshape(2, 3)
        shift = (xyz_max + xyz_min)/2
        enlarge_factor = 1.05
        scale = (xyz_max - xyz_min).max() / 2. * enlarge_factor # enlarge a little
        return shift, scale
        
    def read_intrinsics(self):
        if 'Synthetic' in self.root_dir or 'Ignatius' in self.root_dir:
            with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
                fx = fy = float(f.readline().split()[0]) * self.downsample
            if 'Synthetic' in self.root_dir:
                w = h = int(800*self.downsample)
            else:
                w, h = int(1920*self.downsample), int(1080*self.downsample)

            # K = np.float32([[fx, 0, w/2],
            #                 [0, fy, h/2],
            #                 [0,  0,   1]])
        else:
            K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                           dtype=np.float32)[:3, :3]
            if 'Tanks' in self.root_dir:
                w, h = int(1920*self.downsample), int(1080*self.downsample)
            else:
                assert "error here"
            K[:2] *= self.downsample
            fx = float(K[0, 0])
            fy = float(K[1, 1])

        # self.K = torch.FloatTensor(K)
        # self.img_wh = (w, h)
        cx, cy = w / 2., h / 2.
        return fx, fy, cx, cy
    
    def read_meta(self, split):
        poses = []

        if split == 'test_traj': # BlendedMVS and TanksAndTemple
            assert "should not be here"
            if 'Ignatius' in self.root_dir:
                poses_path = \
                    sorted(glob.glob(os.path.join(self.root_dir, 'test_pose/*.txt')))
                poses = [np.loadtxt(p) for p in poses_path]
            else:
                poses = np.loadtxt(os.path.join(self.root_dir, 'test_traj.txt'))
                poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                c2w = pose[:3]
                c2w[:, 0] *= -1 # [left down front] to [right down front]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                poses += [c2w]
        else:
            if split == 'train': prefix = '0_'
            elif split == 'trainval': prefix = '[0-1]_'
            elif split == 'trainvaltest': prefix = '[0-2]_'
            elif split == 'val': prefix = '1_'
            elif 'Synthetic' in self.root_dir: prefix = '2_' # test set for synthetic scenes
            elif split == 'test': prefix = '1_' # test set for real scenes
            else: raise ValueError(f'{split} split not recognized!')
            img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
            poses_path = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))

            print(f'Loading {len(img_paths)} {split} images ...')
            for pose_path in poses_path:
                try:
                    c2w = np.loadtxt(pose_path)[:3]
                except Exception as e:
                    print(f"Error loading pose: {e}")  
                    import pdb;pdb.set_trace()  
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                poses += [c2w]
        poses = torch.FloatTensor(poses) # (N_images, 3, 4)
        nsvf_to_world = poses
        blender_to_nsvf = torch.tensor([[1, 0, 0, 0], 
                                        [0, -1, 0, 0], 
                                        [0, 0, -1, 0],
                                        [0, 0, 0, 1]], dtype=torch.float)
        B = nsvf_to_world.shape[0]
        blender_to_world = torch.bmm(nsvf_to_world, blender_to_nsvf.unsqueeze(0).repeat(B, 1, 1))
        img_paths = img_paths
        return blender_to_world, img_paths
    
    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        camera_to_world, image_filenames = self.read_meta(split=split)

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor # nothing change here
        scene_box = SceneBox(aabb=torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32))
        
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
        )

        return dataparser_outputs

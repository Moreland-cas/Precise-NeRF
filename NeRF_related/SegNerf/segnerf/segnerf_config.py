from __future__ import annotations
from segnerf.segnerf_model import SegmentNerfConfig, SegmentNerfModel

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.nsvf_dataparser import NsvfDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)

segnerf_config = MethodSpecification(
    config=TrainerConfig(
        method_name="segnerf", 
        max_num_iterations=10**6,
        steps_per_eval_all_images=10**5,
        steps_per_eval_batch=10000,
        steps_per_eval_image=10000,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=BlenderDataParserConfig(),
                # dataparser=NsvfDataParserConfig(alpha_color="white"),
                train_num_rays_per_batch=4096,
            ),
            model=SegmentNerfConfig(
                _target=SegmentNerfModel, 
                background_color = "white", 
                enable_collider=True,
                collider_params = {"near_plane": 2.0, "far_plane": 6.0}, 
                use_mip_tricks=False,
                version="v1"
            )
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-7), 
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, 
                    max_steps=10**6,
                    lr_pre_warmup=5e-6,
                    warmup_steps=2500
                ),
            },
        },
        vis="wandb",
    ),
    description="Segment nerf config",
)

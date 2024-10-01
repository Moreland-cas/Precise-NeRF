from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type, Literal, Optional

import torch
import os
from PIL import Image
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from segnerf.segnerf_field import SegmentNeRFField
from nerfstudio.model_components.losses import MSELoss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider

@dataclass
class SegmentNerfConfig(ModelConfig):
    """Vanilla Model Config"""

    _target: Type = field(default_factory=lambda: SegmentNerfModel)
    num_coarse_samples: int = 64
    """Number of samples in coarse field evaluation"""
    num_importance_samples: int = 128
    """Number of samples in fine field evaluation"""
    enable_collider: bool = True
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = to_immutable_dict({"near_plane": 2.0, "far_plane": 6.0})
    """parameters to instantiate scene collider with"""
    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss_coarse": 1.0,
        "rgb_loss_fine": 1.0,
    })
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color."""
    
    # whether to use the mip tricks
    use_mip_tricks: bool = False
    
    # use v1 version of alpha modelling or v2 version
    version: Literal["v1", "v2"] = "v1"
    

class SegmentNerfModel(Model):
    """Vanilla NeRF model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: SegmentNerfConfig

    def __init__(
        self,
        config: SegmentNerfConfig,
        **kwargs,
    ) -> None:
        self.field_coarse = None
        self.field_fine = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()
        
        # Note: If you are training for NSVF-like dataset, add the following line
        # No need for this line when training for blender-like dataset
        self.collider = AABBBoxCollider(scene_box=self.scene_box)
        
        position_encoding = NeRFEncoding(
            in_dim=6, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        self.field_coarse = SegmentNeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            use_widen_sigmoid=self.config.use_mip_tricks
        )
       
        self.field_fine = SegmentNeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            use_widen_sigmoid=self.config.use_mip_tricks
        )
        
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(
            num_samples=self.config.num_importance_samples, 
            include_original=True, 
            use_smoothing=self.config.use_mip_tricks)
        
        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        self.lpips_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field_coarse.parameters()) + list(self.field_fine.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")
        # choose the wright sampler based on training status
        outputs = {}

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        field_outputs_coarse = self.field_coarse.forward(ray_samples_uniform)
        if self.config.use_gradient_scaling:
            field_outputs_coarse = scale_gradients_by_distance_squared(field_outputs_coarse, ray_samples_uniform)

        weights_coarse = ray_samples_uniform.get_weights(
            field_outputs_coarse[FieldHeadNames.DENSITY],
            version=self.config.version
        )
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # fine field:
        field_outputs_fine = self.field_fine.forward(ray_samples_pdf)
        if self.config.use_gradient_scaling:
            field_outputs_fine = scale_gradients_by_distance_squared(field_outputs_fine, ray_samples_pdf)

        weights_fine = ray_samples_pdf.get_weights(
            field_outputs_fine[FieldHeadNames.DENSITY],
            version=self.config.version
        )
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs.update({
            "rgb": rgb_fine,
            "depth": depth_fine,
            "accumulation": accumulation_fine,
            "rgb_coarse": rgb_coarse, # 4096, 3
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse, # 4096, 1
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine, # 4096, 1
        })

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        if ("rgb_coarse" in outputs.keys()):
            device = outputs["rgb_coarse"].device
        else:
            device = outputs["rgb_coarse_merge"].device
        image = batch["image"].to(device)
        loss_dict = {}

        coarse_pred, coarse_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_coarse"],
            pred_accumulation=outputs["accumulation_coarse"],
            gt_image=image,
        )
        fine_pred, fine_image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fine"],
            pred_accumulation=outputs["accumulation_fine"],
            gt_image=image,
        )

        rgb_loss_coarse = self.rgb_loss(coarse_image, coarse_pred)
        rgb_loss_fine = self.rgb_loss(fine_image, fine_pred)

        loss_dict.update({"rgb_loss_coarse": rgb_loss_coarse, "rgb_loss_fine": rgb_loss_fine})

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        assert self.config.collider_params is not None
        if ("rgb_coarse" in outputs.keys()):
            device = outputs["rgb_coarse"].device
        else:
            device = outputs["rgb_coarse_merge"].device
        image_plot = batch["image"].to(device)
        image_plot = self.renderer_rgb.blend_background(image_plot)
        rgb_list = [image_plot]
        acc_list = []
        depth_list = []
        metrics_dict = {}

        rgb_coarse = outputs["rgb_coarse"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        rgb_fine = outputs["rgb_fine"]
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        rgb_list.extend([rgb_coarse, rgb_fine])
        acc_list.extend([acc_coarse, acc_fine])
        depth_list.extend([depth_coarse, depth_fine])

        image = torch.moveaxis(image_plot, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips_vgg = self.lpips_vgg(image, rgb_fine)
        fine_lpips_alex = self.lpips_alex(image, rgb_fine)
        assert isinstance(fine_ssim, torch.Tensor)

        metrics_dict.update({
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr),
            "ssim": float(fine_ssim),
            "lpips_vgg": float(fine_lpips_vgg),
            "lpips_alex": float(fine_lpips_alex),
        })

        combined_rgb = torch.cat(rgb_list, dim=1)
        combined_acc = torch.cat(acc_list, dim=1)
        combined_depth = torch.cat(depth_list, dim=1)
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict
    
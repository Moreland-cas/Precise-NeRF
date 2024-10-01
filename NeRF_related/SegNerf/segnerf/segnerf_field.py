from typing import Dict, Optional, Tuple, Type
from jaxtyping import Float

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    RGBFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field

class WidenSigmoid(nn.Module):
    def __init__(self, epsilon: float = 0.001):
        super(WidenSigmoid, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        return (1 + 2 * self.epsilon) * torch.sigmoid(x) - self.epsilon


class SegmentNeRFField(Field):
    """NeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        # field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,),
        # field_heads: Optional[Tuple[Type[FieldHead]]] = (),
        use_widen_sigmoid: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        is_coarse: bool = False
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.spatial_distortion = spatial_distortion
        self.use_widen_sigmoid = use_widen_sigmoid
        self.is_coarse = is_coarse
        
        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )
        if self.use_widen_sigmoid:
            self.field_heads = nn.ModuleList([RGBFieldHead(activation=nn.Sigmoid())])  
        else:
            self.field_heads = nn.ModuleList([RGBFieldHead(activation=WidenSigmoid())])  
        
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        positions = ray_samples.frustums.get_start_end_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
        encoded_xyz = self.position_encoding(positions)
        base_mlp_out = self.mlp_base(encoded_xyz)
        density = self.field_output_density(base_mlp_out)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs

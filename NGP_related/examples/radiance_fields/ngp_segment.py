"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import Callable, List, Union

import numpy as np
import einops
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
    #  ord: Union[float, int] = float("inf"),
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x


class NGPRadianceField(torch.nn.Module):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        # Turns out rectangle aabb will leads to uneven collision so bad performance.
        # We enforce a cube aabb here.
        center = (aabb[..., :num_dim] + aabb[..., num_dim:]) / 2.0
        size = (aabb[..., num_dim:] - aabb[..., :num_dim]).max()
        aabb = torch.cat([center - size / 2.0, center + size / 2.0], dim=-1)

        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist() # max_resolution = base_resolution * per_level_scale**(n_levels-1)

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    # {"otype": "Identity", "n_bins": 4, "degree": 4},
                ],
            },
        )

        self.mlp_base = MLPWithHashEncoding(
            num_levels = self.n_levels,
            min_res = self.base_resolution,
            max_res = self.max_resolution,
            log2_hashmap_size = self.log2_hashmap_size,
            features_per_level = 2,
            hash_init_scale = 0.001,
            num_layers = 1, 
            geo_feat_dim = self.geo_feat_dim,
            layer_width = 64,
            out_dim = None,
            skip_connections = None,
            activation = nn.ReLU(),
            out_activation = None,
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=(
                (
                    self.direction_encoding.n_output_dims
                    if self.use_viewdirs
                    else 0
                )
                + self.geo_feat_dim
            ),
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def query_density(self, x, return_feat: bool = False):
        x = einops.rearrange(x, 'b (a c) -> b a c', a=2, c=3)
        
        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        x = einops.rearrange(x, 'b a c -> b (a c)', a=2, c=3)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim * 2)) # torch.Size([2097152, 16])
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim]) 
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            # assert (
            #     positions.shape == directions.shape
            # ), f"{positions.shape} v.s. {directions.shape}"
            density, embedding = self.query_density(positions, return_feat=True)
            rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore

class HashEncoding(nn.Module):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 4096,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001, 
    ) -> None:
        super().__init__()
        self.in_dim = 3
        self.num_levels = num_levels
        self.min_res = min_res
        self.features_per_level = features_per_level
        self.hash_init_scale = hash_init_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        self.scalings = torch.floor(min_res * self.growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size 
        self.hash_table = torch.empty(0)
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        """Initialize the torch version of the hash encoding."""
        self.hash_table = torch.rand(size=(self.hash_table_size * self.num_levels, self.features_per_level)) * 2 - 1
        self.hash_table *= self.hash_init_scale 
        self.hash_table = nn.Parameter(self.hash_table)

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor):
        # : Int[Tensor, "*bs num_levels 3"]) -> Shaped[Tensor, "*bs num_levels"]
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed, B L 3
        """
        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0
        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size # B L
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor):
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""
        assert in_tensor.shape[-1] == 3
        
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # B, L, 3
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f
        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor):
        # : Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]
        return self.pytorch_fwd(in_tensor)

class MLPWithHashEncoding(nn.Module):
    """Multilayer perceptron with hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        num_layers: int = 1,
        geo_feat_dim = 15,
        layer_width: int = 64,
        out_dim = None,
        skip_connections = None,
        activation = nn.ReLU(),
        out_activation = None,
    ) -> None:
        super().__init__()
        self.in_dim = 3

        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.features_per_level = features_per_level
        self.hash_init_scale = hash_init_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size
        self.geo_feat_dim = geo_feat_dim

        self.growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1

        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.tcnn_encoding = None
        self.build_nn_modules()
    
    def build_nn_modules(self) -> None:
        """Initialize the torch version of the MLP with hash encoding."""
        self.hash_encoder = HashEncoding(
            num_levels=self.num_levels,
            min_res=self.min_res,
            max_res=self.max_res,
            log2_hashmap_size=self.log2_hashmap_size,
            features_per_level=self.features_per_level,
            hash_init_scale=self.hash_init_scale,
        )
   
        self.mlp = tcnn.Network(
                n_input_dims=self.num_levels*self.features_per_level*2,
                n_output_dims=1 + self.geo_feat_dim, # 16
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.layer_width,
                    "n_hidden_layers": self.num_layers, 
                },
            )
    
    def forward(self, in_tensor):
        # B 6 -> B 16
        tensor_start, tensor_end = torch.chunk(in_tensor, chunks=2, dim=-1) # B 3
        hash_result_start = self.hash_encoder(tensor_start) # B 32
        hash_result_end = self.hash_encoder(tensor_end)
        hash_result = torch.cat([hash_result_start, hash_result_end], dim=-1) # B 64
        out_tensor = self.mlp(hash_result) # B 16
        return out_tensor

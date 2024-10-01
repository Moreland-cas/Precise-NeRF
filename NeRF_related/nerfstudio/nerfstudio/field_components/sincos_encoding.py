import torch.nn as nn
from typing import Literal, Optional, Sequence
from jaxtyping import Float, Int, Shaped
import numpy as np
import torch
from torch import Tensor, nn


class NeRFEncoding(nn.Module):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
    ) -> None:
        super().__init__()

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

    def pytorch_fwd(
        self,
        in_tensor: Float[Tensor, "*bs input_dim"],
    ) -> Float[Tensor, "*bs output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
       
        return encoded_inputs

    def forward(
        self, in_tensor: Float[Tensor, "*bs input_dim"]
    ) -> Float[Tensor, "*bs output_dim"]:
        encoded_inputs = self.pytorch_fwd(in_tensor)
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs
    
if __name__ == "__main__":
    direction_encoding = NeRFEncoding(
        in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
    )
    data = torch.randn(10, 3)
    print(direction_encoding(data).shape)
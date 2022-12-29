import torch.nn as nn
import torch
"""
Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666

Position Encoding similar to the one used by Transformers
NeRF uses to map its continuous input to a higher-dimensional space using high-frequency functions in order 
to help the model in  learning high frequency variations in the data, which leads to sharper models. 
This approach circumvents the bias that neural networks have towards lower frequency functions, 
allowing NeRF to represent sharper details. 
The authors refer to a paper at ICML 2019 for further reading on this phenomenon (On the Spectral Bias of Neural Networks, ICML19 )
"""


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(
            self,
            d_input: int, # The dimension of the input
            n_freqs: int, # The sampling step; the output dim will be d_input + d_input*2*n_freq
            log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(
            self,
            x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

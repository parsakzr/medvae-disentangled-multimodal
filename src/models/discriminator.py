"""
Discriminator module for adversarial training.
Inspired by Stanford MedVAE discriminator architecture.
"""

import torch
import torch.nn as nn
from typing import Optional


class NLayerDiscriminator(nn.Module):
    """Multi-layer discriminator for adversarial training."""

    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_actnorm: bool = False,
    ):
        super().__init__()
        self.n_layers = n_layers

        def norm_layer(planes: int) -> nn.Module:
            if use_actnorm:
                return nn.GroupNorm(32, planes, affine=True)
            else:
                return nn.BatchNorm2d(planes, affine=True)

        kw = 4
        padw = 1
        use_bias = not use_actnorm

        sequence = [
            nn.Conv2d(
                input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=use_bias
            ),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map

        self.main = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator."""
        return self.main(input)

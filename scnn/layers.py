import math

import torch

from .sh import l_max, ls


class SphConv(torch.nn.Module):
    """A spherical convolution layer.

    Attributes:
        weights (torch.nn.Parameter): Learnable filter coefficients.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.register_buffer("ls", ls)
        self.weights = torch.nn.Parameter(
            torch.Tensor(self.c_out, self.c_in, int(l_max / 2) + 1)
        )
        torch.nn.init.uniform_(self.weights, 0, 1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a spherical convolution on the input.

        Args:
            x: Input signals in the spherical harmonics domain in a tensor with
                shape (batch size, number of channels, number of coefficients).

        Returns:
            Convolved signals.
        """
        weights_exp = self.weights[:, :, (self.ls / 2).long()]
        ys = torch.mean(
            2
            * math.pi
            * torch.sqrt(
                4 * math.pi / (2 * self.ls.unsqueeze(0).unsqueeze(0).unsqueeze(0) + 1)
            )
            * x.unsqueeze(1)
            * weights_exp.unsqueeze(0),
            dim=2,
        )
        return ys

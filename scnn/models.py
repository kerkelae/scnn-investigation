import torch

from .layers import SphConv
from .sh import ls, sft, isft


class SCNNModel(torch.nn.Module):
    """A spherical convolutional neural network model.

    Attributes:
        ls (torch.Tensor): Spherical harmonics degrees.
        sft (torch.Tensor): Spherical Fourier transform.
        isft (torch.Tensor): Inverse spherical Fourier transform.
    """

    def __init__(self, c_in: int, n_out: int):
        """Initialize the model layers.

        Args:
            c_in (int): Number of input channels.
            n_out (int): Number of output scalar parameters.
        """
        super().__init__()
        self.register_buffer("ls", ls)
        self.register_buffer("sft", sft)
        self.register_buffer("isft", isft)
        self.conv1 = SphConv(c_in, 16)
        self.conv2 = SphConv(16, 32)
        self.conv3 = SphConv(32, 64)
        self.conv4 = SphConv(64, 32)
        self.conv5 = SphConv(32, 16)
        self.conv6 = SphConv(16, 1)
        self.fc1 = torch.nn.Linear(112, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, n_out)

    def nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Apply leaky ReLU in the signal domain.

        Args:
            x: Input signals.

        Returns:
            Signals after applying the non-linearity.
        """
        return (
            self.sft
            @ torch.nn.functional.leaky_relu(
                self.isft @ x.unsqueeze(-1),
                negative_slope=0.1,
            )
        ).squeeze(-1)

    def global_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Average signals in the spatial domain.

        Args:
            x: Input signals.

        Returns:
            Signals after pooling.
        """
        return torch.mean(self.isft @ x.unsqueeze(-1), dim=2).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass.

        Args:
            x: Input signals.

        Returns:
            Signals after passing through the network.
        """

        # Encoder
        x = self.conv1(x)
        x1 = self.nonlinearity(x)
        x = self.conv2(x1)
        x2 = self.nonlinearity(x)
        x = self.conv3(x2)
        x3 = self.nonlinearity(x)

        # Decoder
        x = self.conv4(x3)
        x = self.nonlinearity(x)
        x = self.conv5(x)
        x = self.nonlinearity(x)
        odfs_sh = self.conv6(x).squeeze(1)[:, 0:45]

        # Fully connected network
        x = torch.cat(
            [
                self.global_pooling(x1),
                self.global_pooling(x2),
                self.global_pooling(x3),
            ],
            dim=1,
        )
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        scalar_params = self.fc3(x)
        return torch.cat([odfs_sh, scalar_params], dim=1)


class MLPModel(torch.nn.Module):
    """A simple multi-layer perceptron model."""

    def __init__(self, n_in: int, n_out: int):
        """Initialize the model layers.

        Args:
            n_in (int): Number of input features.
            n_out (int): Number of output units.
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(n_in, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.fc4 = torch.nn.Linear(512, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass.

        Args:
            x: Input signals.

        Returns:
            Signals after passing through the network.
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.fc4(x)
        return x

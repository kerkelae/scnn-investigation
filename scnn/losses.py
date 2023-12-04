import torch

from scnn.sh import isft


def mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate mean squared error loss.

    Args:
        preds: Predicted values. The shape must be (batch size, 47).
        targets: Target values. The shape must be (batch size, 47).

    Returns:
        Mean squared error.
    """
    return torch.mean(
        (
            torch.cat(
                (
                    (isft.cpu()[:, 0:45] @ preds.cpu()[:, 0:45].unsqueeze(-1)).squeeze(
                        -1
                    ),
                    preds.cpu()[:, 45::],
                ),
                dim=1,
            )
            - torch.cat(
                (
                    (
                        isft.cpu()[:, 0:45] @ targets.cpu()[:, 0:45].unsqueeze(-1)
                    ).squeeze(-1),
                    targets.cpu()[:, 45::],
                ),
                dim=1,
            )
        )
        ** 2
    )

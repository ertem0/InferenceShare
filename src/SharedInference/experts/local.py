"""Local inference for a single bound PyTorch expert."""

import torch
from torch import Tensor, nn


class LocalExpertExecutor:
    """Bind one expert to its layer and expert IDs.

    The supplied module must map [num_tokens, hidden_size] to the same shape.
    This executor sets the module (including its children) to evaluation mode
    on construction and execution. Supply a dedicated inference module if it
    is also needed for training elsewhere. Model computation errors propagate
    unchanged, including incompatible input dtype, device, or hidden width.
    """

    def __init__(self, layer_id: int, expert_id: int, expert: nn.Module) -> None:
        self._identity = (layer_id, expert_id)
        self._expert = expert
        self._expert.eval()

    def execute(self, layer_id: int, expert_id: int, hidden_states: Tensor) -> Tensor:
        """Execute the bound expert, raising on unknown IDs or invalid shapes."""
        if (layer_id, expert_id) != self._identity:
            raise KeyError(
                f"Unknown expert: layer_id={layer_id}, expert_id={expert_id}"
            )
        if not isinstance(hidden_states, Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [num_tokens, hidden_size]")

        self._expert.eval()
        with torch.inference_mode():
            output = self._expert(hidden_states)

        if not isinstance(output, Tensor):
            raise TypeError("Expert output must be a torch.Tensor")
        if output.shape != hidden_states.shape:
            raise ValueError(
                f"Expert output shape {tuple(output.shape)} does not match "
                f"input shape {tuple(hidden_states.shape)}"
            )
        return output

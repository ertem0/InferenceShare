"""Common contract for local and future remote expert executors."""

from typing import Protocol

from torch import Tensor


class ExpertExecutor(Protocol):
    """Execute an identified expert without routing or weighting its output."""

    def execute(self, layer_id: int, expert_id: int, hidden_states: Tensor) -> Tensor:
        """Return expert outputs with shape [num_tokens, hidden_size].

        Inputs have the same shape. Execution is inference-only. Callers supply
        tensors with the dtype and device required by the expert; executors do
        not implicitly move or cast them. Unknown identities raise KeyError.
        """
        ...

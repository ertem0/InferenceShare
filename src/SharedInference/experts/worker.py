"""Local expert hosting, independent of checkpoint loading and transport."""

from torch import Tensor

from .identified import IdentifiedExpert
from .local import LocalExpertExecutor


class ExpertWorker:
    """Host already constructed expert modules on one node.

    Initialization code supplies modules through register(). In milestone 5,
    a receiver will reconstruct them from coordinator-supplied weights before
    registration. This worker performs no downloads, checkpoint reads, weight
    reconstruction, or remote calls. Register experts before serving requests.
    """

    def __init__(self) -> None:
        self._executors: dict[tuple[int, int], LocalExpertExecutor] = {}

    def register(self, expert: IdentifiedExpert) -> None:
        """Accept an identified local module from initialization; reject duplicate identities.

        This is the handoff point for the future receiver. The module is held
        by reference and set to evaluation mode by LocalExpertExecutor.
        """
        if not isinstance(expert, IdentifiedExpert):
            raise TypeError("expert must be an IdentifiedExpert")
        layer_id, expert_id = expert.layer_id, expert.expert_id
        identity = (layer_id, expert_id)
        if identity in self._executors:
            raise ValueError(
                f"Expert already registered: layer_id={layer_id}, expert_id={expert_id}"
            )
        self._executors[identity] = LocalExpertExecutor(expert)

    def execute(self, layer_id: int, expert_id: int, hidden_states: Tensor) -> Tensor:
        """Execute a registered expert through the common executor interface."""
        identity = (layer_id, expert_id)
        if identity not in self._executors:
            raise KeyError(
                f"Unknown expert: layer_id={layer_id}, expert_id={expert_id}"
            )
        return self._executors[identity].execute(layer_id, expert_id, hidden_states)

"""Expert identity carried alongside its local computation module."""

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class IdentifiedExpert:
    """Bind immutable identity metadata to an already constructed module.

    Freezing protects the identity fields, not the module's weights or mode.
    Receivers reconstruct this container after receiving metadata and weights;
    it is not a network serialization format.
    """

    layer_id: int
    expert_id: int
    module: nn.Module

    def __post_init__(self) -> None:
        for name, value in (("layer_id", self.layer_id), ("expert_id", self.expert_id)):
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not isinstance(self.module, nn.Module):
            raise TypeError("module must be a local torch.nn.Module")

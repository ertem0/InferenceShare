"""OLMoE startup adapter: checkpoint inventory and fresh expert reconstruction."""

import json
from pathlib import Path

import torch

from SharedInference.experts import IdentifiedExpert

from .olmoe import OlmoeExpert, load_olmoe_expert

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class OlmoeExpertSource:
    """Inspect model configuration without loading all expert weights.

    Startup chooses one explicit CPU dtype (float32 by default) so all experts
    have a known, equal memory cost before allocation.
    """

    def __init__(self, checkpoint_dir, *, dtype=torch.float32):
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        if dtype not in _DTYPES.values():
            raise ValueError("Startup dtype must be float32, float16, or bfloat16")
        self.dtype = dtype
        config = json.loads((self.checkpoint_dir / "config.json").read_text())
        if config.get("model_type") != "olmoe" or config.get("hidden_act") != "silu":
            raise ValueError("Expected an OLMoE checkpoint with SiLU activation")
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_experts",
        ):
            if type(config.get(name)) is not int or config[name] <= 0:
                raise ValueError(f"Invalid positive integer config field: {name}")
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.expert_ids = tuple(
            (layer, expert)
            for layer in range(config["num_hidden_layers"])
            for expert in range(config["num_experts"])
        )
        self.expert_size_bytes = (
            3
            * self.hidden_size
            * self.intermediate_size
            * torch.empty((), dtype=dtype).element_size()
        )

    def prepare(self, layer_id, expert_id):
        expert = load_olmoe_expert(
            self.checkpoint_dir, layer_id, expert_id, dtype=self.dtype
        )
        return {
            "model_type": "olmoe",
            "hidden_act": "silu",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "dtype": str(self.dtype).removeprefix("torch."),
        }, expert.module.state_dict()


def reconstruct_olmoe_expert(layer_id, expert_id, metadata, weights):
    """Validate received weights and build a fresh module without checkpoint access."""
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_type") != "olmoe"
        or metadata.get("hidden_act") != "silu"
    ):
        raise ValueError("Incompatible OLMoE reconstruction metadata")
    for name in ("hidden_size", "intermediate_size"):
        if type(metadata.get(name)) is not int or metadata[name] <= 0:
            raise ValueError(f"Invalid positive integer metadata field: {name}")
    dtype = _DTYPES.get(metadata.get("dtype"))
    if dtype is None:
        raise ValueError("Unsupported expert dtype")
    hidden, intermediate = metadata["hidden_size"], metadata["intermediate_size"]
    shapes = {
        "gate_proj.weight": (intermediate, hidden),
        "up_proj.weight": (intermediate, hidden),
        "down_proj.weight": (hidden, intermediate),
    }
    if weights.keys() != shapes.keys():
        raise ValueError("Unexpected expert weight names")
    for name, shape in shapes.items():
        tensor = weights[name]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device.type != "cpu"
        ):
            raise ValueError(f"Incompatible expert weight: {name}")
    with torch.device("meta"):
        expert = OlmoeExpert(hidden, intermediate)
    expert.load_state_dict(weights, strict=True, assign=True)
    return IdentifiedExpert(layer_id, expert_id, expert.eval().requires_grad_(False))

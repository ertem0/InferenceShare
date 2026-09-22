"""Offline loading of individual experts from indexed OLMoE checkpoints."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional as F

from SharedInference.experts.identified import IdentifiedExpert


class OlmoeExpert(nn.Module):
    """Bias-free SiLU gated MLP used by the OLMoE checkpoint."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def load_olmoe_expert(
    checkpoint_dir: str | Path,
    layer_id: int,
    expert_id: int,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device = "cpu",
) -> IdentifiedExpert:
    """Load only three selected tensors, preserving checkpoint dtype by default.

    Returns identity metadata alongside the module.
    No Hub access or full-model construction occurs. Only indexed checkpoints
    with separate gate/up/down expert weights are supported. Returned weights
    own their storage and do not retain shard memory mappings. Missing files
    raise FileNotFoundError; invalid config/weights raise ValueError; unknown
    identities or missing tensor keys raise KeyError.
    """
    root = Path(checkpoint_dir).expanduser().resolve()
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "olmoe" or config.get("hidden_act") != "silu":
        raise ValueError("Expected an OLMoE checkpoint with hidden_act='silu'")
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_experts",
    ):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"Invalid positive integer config field: {name}")
    if (
        type(layer_id) is not int
        or type(expert_id) is not int
        or not 0 <= layer_id < config["num_hidden_layers"]
        or not 0 <= expert_id < config["num_experts"]
    ):
        raise KeyError(f"Unknown expert: layer_id={layer_id}, expert_id={expert_id}")
    if dtype is not None and dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError("Expert dtype must be floating point")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Checkpoint index must contain a weight_map object")  # noqa: TRY004
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    shapes = {
        "gate_proj.weight": (intermediate, hidden),
        "up_proj.weight": (intermediate, hidden),
        "down_proj.weight": (hidden, intermediate),
    }
    prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}."
    weights = {}
    source_dtype = None
    for name, shape in shapes.items():
        key = prefix + name
        if key not in weight_map:
            raise KeyError(f"Missing expert weight in checkpoint index: {key}")
        filename = weight_map[key]
        if not isinstance(filename, str):
            raise ValueError(f"Invalid shard filename for {key}")  # noqa: TRY004
        shard = (root / filename).resolve()
        if not shard.is_relative_to(root):
            raise ValueError(f"Shard path escapes checkpoint directory: {filename}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if key not in handle.keys():  # noqa: SIM118 - safe_open is not a mapping
                raise KeyError(f"Missing expert weight in {filename}: {key}")
            if tuple(handle.get_slice(key).get_shape()) != shape:
                raise ValueError(f"Unexpected shape for {key}; expected {shape}")
            tensor = handle.get_tensor(key)
            if not tensor.is_floating_point():
                raise ValueError(f"Expected floating-point weight: {key}")
            if source_dtype is not None and tensor.dtype != source_dtype:
                raise ValueError("Expert weights have inconsistent dtypes")
            source_dtype = tensor.dtype
            weights[name] = tensor.to(
                device=device, dtype=dtype or tensor.dtype, copy=True
            )
    # Meta construction avoids allocating or initializing an extra set of weights.
    with torch.device("meta"):
        expert = OlmoeExpert(hidden, intermediate)
    expert.load_state_dict(weights, strict=True, assign=True)
    return IdentifiedExpert(layer_id, expert_id, expert.eval().requires_grad_(False))

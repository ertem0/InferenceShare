import pytest
import torch

from SharedInference.model.olmoe_initialization import (
    OlmoeExpertSource,
    reconstruct_olmoe_expert,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_source_inventory_size_and_fresh_reconstruction(olmoe_checkpoint, dtype):
    source = OlmoeExpertSource(olmoe_checkpoint[0], dtype=dtype)
    assert source.expert_ids == ((0, 0), (0, 1), (1, 0), (1, 1))
    metadata, weights = source.prepare(1, 0)
    assert source.expert_size_bytes == sum(
        t.numel() * t.element_size() for t in weights.values()
    )
    expert = reconstruct_olmoe_expert(1, 0, metadata, weights)
    assert (expert.layer_id, expert.expert_id) == (1, 0)
    assert not expert.module.training
    assert all(not p.requires_grad for p in expert.module.parameters())


@pytest.mark.parametrize("mutation", ["shape", "dtype", "missing", "metadata"])
def test_reconstruction_rejects_incompatible_weights(olmoe_checkpoint, mutation):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    metadata, weights = source.prepare(0, 0)
    if mutation == "shape":
        weights["gate_proj.weight"] = torch.zeros(2, 3)
    elif mutation == "dtype":
        weights["gate_proj.weight"] = weights["gate_proj.weight"].half()
    elif mutation == "missing":
        del weights["up_proj.weight"]
    else:
        metadata["model_type"] = "unknown"
    with pytest.raises(ValueError):
        reconstruct_olmoe_expert(0, 0, metadata, weights)

"""Reference tests are optional; real weights are never downloaded by pytest."""

import json
import os
from pathlib import Path

import pytest
import torch

from SharedInference.experts import LocalExpertExecutor
from SharedInference.model.olmoe import load_olmoe_expert


def compare_reference(root, layer_id, expert_id):
    pytest.importorskip("transformers")
    from safetensors import safe_open
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

    config = OlmoeConfig(**json.loads((root / "config.json").read_text()))
    reference = OlmoeMLP(config).eval()
    # Read reference weights independently of the adapter under test.
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = {}
    for name in reference.state_dict():
        key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{name}"
        with safe_open(root / index[key], framework="pt", device="cpu") as handle:
            weights[name] = handle.get_tensor(key).float().clone()
    reference.load_state_dict(weights)
    expert = load_olmoe_expert(root, layer_id, expert_id, dtype=torch.float32)
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(4, config.hidden_size, generator=generator)
    with torch.inference_mode():
        expected = reference(inputs)
    actual = LocalExpertExecutor(layer_id, expert_id, expert).execute(
        layer_id, expert_id, inputs
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fixture_matches_transformers(olmoe_checkpoint):
    compare_reference(olmoe_checkpoint[0], 1, 1)


@pytest.mark.skipif(
    not os.environ.get("OLMOE_CHECKPOINT_DIR"),
    reason="Set OLMOE_CHECKPOINT_DIR to a local real checkpoint",
)
def test_real_checkpoint_matches_transformers():
    compare_reference(Path(os.environ["OLMOE_CHECKPOINT_DIR"]).expanduser(), 0, 0)

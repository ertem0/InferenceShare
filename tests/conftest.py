import json

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def olmoe_checkpoint(tmp_path):
    config = {
        "model_type": "olmoe",
        "hidden_act": "silu",
        "hidden_size": 2,
        "intermediate_size": 3,
        "num_hidden_layers": 2,
        "num_experts": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    shards = [{}, {}]
    weight_map, experts = {}, {}
    for layer in range(2):
        for expert in range(2):
            weights = {}
            for i, (name, shape) in enumerate(
                [
                    ("gate_proj.weight", (3, 2)),
                    ("up_proj.weight", (3, 2)),
                    ("down_proj.weight", (2, 3)),
                ]
            ):
                tensor = (torch.arange(6, dtype=torch.float32).reshape(shape) - 2) / 10
                tensor = tensor + layer * 0.2 + expert * 0.1 + i * 0.05
                key = f"model.layers.{layer}.mlp.experts.{expert}.{name}"
                shards[i % 2][key] = tensor
                weight_map[key] = f"shard-{i % 2}.safetensors"
                weights[name] = tensor
            experts[layer, expert] = weights
    for i, tensors in enumerate(shards):
        save_file(tensors, tmp_path / f"shard-{i}.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return tmp_path, config, experts

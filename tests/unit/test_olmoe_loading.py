import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from SharedInference.experts import LocalExpertExecutor
from SharedInference.model.olmoe import load_olmoe_expert


@pytest.mark.parametrize("identity", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_selects_expert_across_shards(olmoe_checkpoint, identity):
    root, _, experts = olmoe_checkpoint
    weights = experts[identity]
    expert = load_olmoe_expert(root, *identity)
    x = torch.tensor([[1.0, -2.0], [0.5, 1.0]])
    gate = x @ weights["gate_proj.weight"].T
    expected = (
        (gate * torch.sigmoid(gate)) * (x @ weights["up_proj.weight"].T)
    ) @ weights["down_proj.weight"].T
    executor = LocalExpertExecutor(*identity, expert)
    for _ in range(2):
        torch.testing.assert_close(executor.execute(*identity, x), expected)
    assert not expert.training
    assert all(not p.requires_grad for p in expert.parameters())


def test_only_requested_tensors_are_read(olmoe_checkpoint, monkeypatch):
    from SharedInference.model import olmoe

    root, _, _ = olmoe_checkpoint
    original, reads = olmoe.safe_open, []

    class TrackedFile:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)

        def __enter__(self):
            self.handle = self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, key):
            return self.handle.get_slice(key)

        def get_tensor(self, key):
            reads.append(key)
            return self.handle.get_tensor(key)

    monkeypatch.setattr(olmoe, "safe_open", TrackedFile)
    load_olmoe_expert(root, 1, 0)
    assert len(reads) == 3
    assert all(k.startswith("model.layers.1.mlp.experts.0.") for k in reads)


@pytest.mark.parametrize("identity", [(-1, 0), (2, 0), (0, 2)])
def test_unknown_expert(olmoe_checkpoint, identity):
    with pytest.raises(KeyError, match="Unknown expert"):
        load_olmoe_expert(olmoe_checkpoint[0], *identity)


def test_explicit_dtype_conversion(olmoe_checkpoint):
    expert = load_olmoe_expert(olmoe_checkpoint[0], 0, 0, dtype=torch.float64)
    assert all(p.dtype == torch.float64 for p in expert.parameters())


def test_missing_shard(olmoe_checkpoint):
    root = olmoe_checkpoint[0]
    (root / "shard-0.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        load_olmoe_expert(root, 0, 0)


@pytest.mark.parametrize(
    "fault", ["missing_key", "shape", "dtype", "mixed_dtype", "path"]
)
def test_invalid_weights(olmoe_checkpoint, fault):
    root = olmoe_checkpoint[0]
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    if fault == "missing_key":
        del index["weight_map"][key]
    elif fault == "path":
        index["weight_map"][key] = "../outside.safetensors"
    else:
        path = root / "shard-0.safetensors"
        tensors = load_file(path)
        tensors[key] = {
            "shape": torch.zeros(1, 1),
            "dtype": torch.zeros(3, 2, dtype=torch.int32),
            "mixed_dtype": torch.zeros(3, 2, dtype=torch.float64),
        }[fault]
        save_file(tensors, path)
    index_path.write_text(json.dumps(index))
    with pytest.raises(KeyError if fault == "missing_key" else ValueError):
        load_olmoe_expert(root, 0, 0)


def test_invalid_configuration(olmoe_checkpoint):
    root, config, _ = olmoe_checkpoint
    config["hidden_act"] = "gelu"
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hidden_act"):
        load_olmoe_expert(root, 0, 0)

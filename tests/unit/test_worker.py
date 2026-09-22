import pytest
import torch
from torch import nn

from SharedInference.experts import (
    ExpertExecutor,
    ExpertWorker,
    IdentifiedExpert,
    LocalExpertExecutor,
)


def scaled_expert(scale):
    expert = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        expert.weight.copy_(torch.eye(2) * scale)
    return expert


def test_selects_experts_by_layer_and_expert():
    worker = ExpertWorker()
    for layer, expert, scale in [(0, 0, 2), (0, 1, 3), (1, 0, 4)]:
        worker.register(IdentifiedExpert(layer, expert, scaled_expert(scale)))
    executor: ExpertExecutor = worker
    inputs = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    for _ in range(2):
        for layer, expert, scale in [(0, 0, 2), (0, 1, 3), (1, 0, 4)]:
            torch.testing.assert_close(
                executor.execute(layer, expert, inputs), inputs * scale
            )


@pytest.mark.parametrize("identity", [(0, 1), (1, 0), (1, 1)])
def test_unknown_identity(identity):
    worker = ExpertWorker()
    worker.register(IdentifiedExpert(0, 0, scaled_expert(2)))
    with pytest.raises(KeyError, match="Unknown expert"):
        worker.execute(*identity, torch.ones(1, 2))


def test_empty_worker_can_be_initialized_later():
    worker = ExpertWorker()
    with pytest.raises(KeyError, match="Unknown expert"):
        worker.execute(0, 0, torch.ones(1, 2))
    worker.register(IdentifiedExpert(0, 0, scaled_expert(2)))
    torch.testing.assert_close(
        worker.execute(0, 0, torch.ones(1, 2)), torch.full((1, 2), 2.0)
    )


def test_duplicate_registration_preserves_original():
    worker = ExpertWorker()
    worker.register(IdentifiedExpert(0, 0, scaled_expert(2)))
    replacement = scaled_expert(3)
    with pytest.raises(ValueError, match="already registered"):
        worker.register(IdentifiedExpert(0, 0, replacement))
    assert replacement.training
    torch.testing.assert_close(
        worker.execute(0, 0, torch.ones(1, 2)), torch.full((1, 2), 2.0)
    )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        "checkpoint/path",
        LocalExpertExecutor(IdentifiedExpert(0, 0, nn.Identity())),
    ],
)
def test_registration_requires_local_module(invalid):
    worker = ExpertWorker()
    with pytest.raises(TypeError, match="IdentifiedExpert"):
        worker.register(invalid)
    worker.register(IdentifiedExpert(0, 0, scaled_expert(2)))


def test_inference_mode_and_validation_are_preserved():
    class InferenceExpert(nn.Module):
        def forward(self, inputs):
            assert not self.training
            assert torch.is_inference_mode_enabled()
            return inputs * 2

    expert = InferenceExpert()
    worker = ExpertWorker()
    worker.register(IdentifiedExpert(0, 0, expert))
    assert not expert.training
    result = worker.execute(0, 0, torch.ones(1, 2, requires_grad=True))
    assert not result.requires_grad
    with pytest.raises(ValueError, match="hidden_states must have shape"):
        worker.execute(0, 0, torch.ones(2))

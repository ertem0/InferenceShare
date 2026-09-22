import pytest
import torch
from torch import nn

from SharedInference.experts import (
    ExpertExecutor,
    IdentifiedExpert,
    LocalExpertExecutor,
)


class SmallExpert(nn.Module):
    """Fixed-weight feed-forward expert with hand-computable outputs."""

    def __init__(self):
        super().__init__()
        self.up = nn.Linear(2, 3)
        self.down = nn.Linear(3, 2)
        with torch.no_grad():
            self.up.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]]))
            self.up.bias.copy_(torch.tensor([0.0, 1.0, 0.0]))
            self.down.weight.copy_(torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 2.0]]))
            self.down.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, hidden_states):
        return self.down(torch.relu(self.up(hidden_states)))


def test_selected_expert_matches_expected_values_repeatedly():
    executor: ExpertExecutor = LocalExpertExecutor(
        IdentifiedExpert(3, 7, SmallExpert())
    )
    inputs = torch.tensor([[2.0, 1.0], [-1.0, 2.0], [0.0, -2.0]])
    expected = torch.tensor([[5.5, 2.5], [5.5, 10.5], [0.5, -0.5]])
    original = inputs.clone()

    for _ in range(2):
        actual = executor.execute(3, 7, inputs)
        assert actual.shape == inputs.shape
        assert actual.dtype == inputs.dtype
        assert actual.device == inputs.device
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(inputs, original)


@pytest.mark.parametrize("layer_id, expert_id", [(4, 7), (3, 8), (4, 8)])
def test_unknown_identity_fails_before_execution(layer_id, expert_id):
    class MustNotRun(nn.Module):
        def forward(self, hidden_states):
            pytest.fail("Unknown identities must not execute the expert")

    executor = LocalExpertExecutor(IdentifiedExpert(3, 7, MustNotRun()))
    with pytest.raises(KeyError, match="Unknown expert"):
        executor.execute(layer_id, expert_id, torch.ones(1, 2))


def test_execution_uses_eval_and_inference_mode():
    class ModeExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=1.0)

        def forward(self, hidden_states):
            assert torch.is_inference_mode_enabled()
            assert not self.training
            assert not self.dropout.training
            return self.dropout(hidden_states) * 2

    expert = ModeExpert()
    executor = LocalExpertExecutor(IdentifiedExpert(0, 0, expert))
    assert not expert.training
    expert.train()
    output = executor.execute(0, 0, torch.ones(2, 3, requires_grad=True))
    torch.testing.assert_close(output, torch.full((2, 3), 2.0))
    assert not output.requires_grad


@pytest.mark.parametrize("shape", [(), (2,), (1, 2, 3)])
def test_invalid_input_rank(shape):
    executor = LocalExpertExecutor(IdentifiedExpert(0, 0, nn.Identity()))
    with pytest.raises(ValueError, match="hidden_states must have shape"):
        executor.execute(0, 0, torch.zeros(shape))


def test_invalid_input_type():
    executor = LocalExpertExecutor(IdentifiedExpert(0, 0, nn.Identity()))
    with pytest.raises(TypeError, match="hidden_states must be a torch.Tensor"):
        executor.execute(0, 0, [[1.0, 2.0]])


def test_invalid_output_shape():
    executor = LocalExpertExecutor(IdentifiedExpert(0, 0, nn.Linear(2, 3)))
    with pytest.raises(ValueError, match="Expert output shape"):
        executor.execute(0, 0, torch.ones(1, 2))


def test_invalid_output_type():
    class TupleExpert(nn.Module):
        def forward(self, hidden_states):
            return (hidden_states,)

    executor = LocalExpertExecutor(IdentifiedExpert(0, 0, TupleExpert()))
    with pytest.raises(TypeError, match="Expert output must be a torch.Tensor"):
        executor.execute(0, 0, torch.ones(1, 2))


def test_model_errors_propagate():
    executor = LocalExpertExecutor(IdentifiedExpert(0, 0, SmallExpert()))
    with pytest.raises(RuntimeError):
        executor.execute(0, 0, torch.ones(1, 4))


def test_empty_token_batch():
    executor = LocalExpertExecutor(IdentifiedExpert(0, 0, SmallExpert()))
    assert executor.execute(0, 0, torch.empty(0, 2)).shape == (0, 2)

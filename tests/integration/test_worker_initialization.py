"""Exercise the registration handoff with modules prepared outside the worker."""

import torch
from torch import nn

from SharedInference.experts import ExpertWorker, IdentifiedExpert


def test_external_initialization_registers_prepared_modules():
    # Stand in for a future receiver after it reconstructs coordinator weights.
    # There is no transport or worker-side checkpoint loading in this milestone.
    worker = ExpertWorker()
    for expert_id, weight in enumerate(
        [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        ]
    ):
        module = nn.Linear(2, 2, bias=False)
        module.load_state_dict({"weight": weight})
        worker.register(IdentifiedExpert(0, expert_id, module))
    inputs = torch.tensor([[1.0, 2.0]])
    torch.testing.assert_close(
        worker.execute(0, 0, inputs), torch.tensor([[5.0, 11.0]])
    )
    torch.testing.assert_close(worker.execute(0, 1, inputs), torch.tensor([[2.0, 6.0]]))


def test_checkpoint_identity_flows_into_registration(olmoe_checkpoint):
    from SharedInference.model.olmoe import load_olmoe_expert

    prepared = load_olmoe_expert(olmoe_checkpoint[0], 1, 0)
    worker = ExpertWorker()
    worker.register(prepared)
    inputs = torch.ones(2, 2)
    with torch.inference_mode():
        expected = prepared.module(inputs)
    torch.testing.assert_close(worker.execute(1, 0, inputs), expected)

from dataclasses import FrozenInstanceError

import pytest
from torch import nn

from SharedInference.experts import IdentifiedExpert


def test_identity_is_immutable_and_module_retained():
    module = nn.Identity()
    expert = IdentifiedExpert(1, 2, module)
    assert expert.module is module
    with pytest.raises(FrozenInstanceError):
        expert.expert_id = 3


@pytest.mark.parametrize("layer, expert", [(-1, 0), (0, -1)])
def test_negative_identity(layer, expert):
    with pytest.raises(ValueError, match="nonnegative"):
        IdentifiedExpert(layer, expert, nn.Identity())


@pytest.mark.parametrize("layer, expert", [(True, 0), (0, 1.0), ("0", 0)])
def test_noninteger_identity(layer, expert):
    with pytest.raises(TypeError, match="integer"):
        IdentifiedExpert(layer, expert, nn.Identity())


def test_requires_module():
    with pytest.raises(TypeError, match="local torch.nn.Module"):
        IdentifiedExpert(0, 0, {})

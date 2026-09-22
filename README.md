# Distributed MoE Inference

A research prototype for distributing Mixture-of-Experts inference across multiple devices.

Different nodes store different experts. During inference, hidden states are routed to the node containing the selected expert, computed remotely, and returned to the main inference process.

## Status

Early research prototype. Milestone 1 local execution is implemented with
offline correctness tests. Milestone 2 includes checkpoint downloading and
selective OLMoE expert loading. Milestone 3 adds local multi-expert workers.
Real-checkpoint validation requires a local download.

See [PROJECT.md](PROJECT.md) for the current scope and milestones.

## Requirements

- Python
- PyTorch

## Installation

```bash
uv sync
```

## Usage

Bind a PyTorch expert to a layer and expert ID, then execute it locally:

```python
import torch
from torch import nn

from SharedInference.experts import IdentifiedExpert, LocalExpertExecutor

expert = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
executor = LocalExpertExecutor(IdentifiedExpert(layer_id=0, expert_id=3, module=expert))
outputs = executor.execute(0, 3, torch.ones(2, 4))
assert outputs.shape == (2, 4)
```

Experts accept and return tensors shaped `[num_tokens, hidden_size]`. The
executor sets the supplied module to evaluation mode and runs it under
`torch.inference_mode()`. Inputs must already match the expert's dtype and
device; no automatic conversion is performed. Unknown layer/expert pairs raise
`KeyError`. Invalid tensor types or shapes raise `TypeError` or `ValueError`;
errors from the underlying module propagate unchanged.

The model-independent `ExpertExecutor` protocol defines the common execution
interface. Each local executor binds one expert. `ExpertWorker` implements the
same interface for multiple registered local experts. Routing and networking
belong to later milestones.

Run the offline tests and code checks with:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Host experts in a worker

```python
from SharedInference.experts import ExpertWorker, IdentifiedExpert

worker = ExpertWorker()
# Initialization supplies an already constructed local PyTorch module.
worker.register(IdentifiedExpert(layer_id=0, expert_id=3, module=expert))
outputs = worker.execute(0, 3, torch.ones(2, 4))
```

This example uses the four-dimensional expert from the first usage example.
Identity fields in `IdentifiedExpert` are immutable; the contained module remains mutable.
Workers start empty. `register()` is the handoff point for the future receiver,
which will reconstruct modules from weights sent by the coordinator in
milestone 5. Workers do not load checkpoints, download weights, or host remote
executors. They retain supplied modules in memory for local execution.
Register experts before serving requests; concurrent registration is not supported.
Duplicate identities raise `ValueError`, unknown identities raise `KeyError`,
and execution uses the existing local executor's inference and validation behavior.

## Download and load OLMoE experts

Download the pinned checkpoint directly to your internal SSD:

```bash
uv run python -m SharedInference.model.checkpoint
```

The default destination on Vasco's machine is:

```text
/Users/vasco/Models/allenai/OLMoE-1B-7B-0924/6d84c48581ece794365f2b8e9cfb043c68ade9c5/
```

On other machines the base directory is `~/Models`. Use `--destination PATH`
to override it. For another checkpoint, pass both `--model-id` and `--revision`
(the full commit hash). The downloader retrieves configuration, index, and all
Safetensors shards, using one concurrent file download. It checks available
space first and reuses completed downloads. Allow approximately 14 GB plus
headroom for OLMoE. Downloads use memory buffers; they never construct the model.
The preflight check contacts Hugging Face; local expert loading does not.

Load a single expert from disk and execute it:

```python
from pathlib import Path
import torch

from SharedInference.experts import LocalExpertExecutor
from SharedInference.model.checkpoint import OLMOE_MODEL_ID, OLMOE_REVISION
from SharedInference.model.olmoe import load_olmoe_expert

checkpoint = Path.home() / "Models" / OLMOE_MODEL_ID / OLMOE_REVISION
expert = load_olmoe_expert(checkpoint, layer_id=0, expert_id=3, dtype=torch.float32)
executor = LocalExpertExecutor(expert)
outputs = executor.execute(0, 3, torch.ones(2, expert.module.gate_proj.in_features))
```

The loader reads only the selected expert's three tensors, including when they
span shards. It returns an `IdentifiedExpert` containing the selected IDs and an evaluation-mode
module with gradients disabled. Pass this object directly to `worker.register(expert)`;
registration does not require repeating the IDs.
By default it preserves the stored dtype on CPU; `dtype` and `device` provide
explicit conversion. It supports indexed OLMoE checkpoints with separate
SiLU gate/up/down projection weights. Missing files, unknown IDs, invalid
metadata, and incompatible tensor shapes or dtypes raise explicit errors.

Default tests use tiny local checkpoint fixtures and mocked download calls.
To additionally compare against the pinned Transformers reference:

```bash
uv run --group reference pytest
```

To validate the real checkpoint after downloading it (this test never downloads):

```bash
OLMOE_CHECKPOINT_DIR="$HOME/Models/allenai/OLMoE-1B-7B-0924/6d84c48581ece794365f2b8e9cfb043c68ade9c5" \
    uv run --group reference pytest tests/integration/test_olmoe_reference.py
```

Node initialization and expert distribution remain in milestone 5.

## Project structure

```text
src/SharedInference/experts/   Execution protocol, local executor, and worker
src/SharedInference/model/   Checkpoint download and OLMoE adapter
tests/unit/                  Deterministic execution and loading tests
tests/integration/           Optional Transformers reference comparisons
```

## Documentation

- [Project scope](PROJECT.md)
- [Architecture](ARCHITECTURE.md)

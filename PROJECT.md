# Distributed MoE Inference Prototype

## Objective

Build a prototype that distributes MoE experts across multiple machines.

Each node stores only a subset of the model's experts.

The system should allow an inference coordinator to:

1. run the shared dense components
2. determine which experts are selected by the router
3. send hidden states to the nodes containing those experts
4. receive expert outputs
5. aggregate the outputs
6. continue inference

The initial goal is experimental correctness and measurable communication
performance, not production-scale inference.

## V1 target

A two-node prototype where:

- Node A runs the main model components.
- Node B owns one or more remote experts.
- Node A sends expert inputs to Node B.
- Node B performs the expert computation.
- Node B returns the result.
- Node A continues the forward pass.

## Milestones

### Milestone 1 — Local Expert Execution

Implement the basic expert execution abstraction without any networking.
For testing lets use allenai/OLMoE-1B-7B-0924 model but the software should not be model specific.

Requirements:

- define an expert module using PyTorch
- support identifying experts by `layer_id` and `expert_id`
- define a common expert execution interface
- allow executing a specific expert locally
- verify the returned tensor shape and values

Acceptance criteria:

- a test can select a specific expert and execute it locally
- repeated execution with the same input and weights produces the expected output
- the expert execution interface does not depend on networking

---

### Milestone 2 — Model Download and Local Expert Loading

Download the model checkpoint from Hugging Face to the coordinator's SSD and
load specific experts from that local checkpoint for execution.

Requirements:

- support a configurable Hugging Face model ID, pinned revision, and local checkpoint directory
- use `allenai/OLMoE-1B-7B-0924` as the first supported model, isolating model-specific weight extraction from download logic
- download the checkpoint to the coordinator's SSD and allow subsequent starts to use that local checkpoint without downloading it again
- load selected expert weights by `(layer_id, expert_id)` without constructing the entire model in memory
- reconstruct a selected expert as a PyTorch module compatible with the local expert execution interface
- fail explicitly on unknown expert identities, missing weights, incompatible metadata, or download failures

Acceptance criteria:

- the checkpoint can be downloaded to a configured local directory and reused on a later startup
- a specific expert can be selected by `(layer_id, expert_id)` and loaded from SSD without loading the entire model into memory
- the loaded expert executes through the local executor and matches reference execution with the same weights and inputs within floating-point tolerance
- expert loading works offline once the required checkpoint files are present
- automated loading tests use small local fixtures without requiring Hugging Face access

Sending weights to worker nodes and initializing remote workers belong to
Milestone 5, after the worker and transport layer are available.

---

### Milestone 3 — Expert Worker

Implement a worker capable of hosting and executing multiple experts locally.

Requirements:

- create an `ExpertWorker`
- register multiple already constructed local PyTorch expert modules supplied by initialization code
- expose registration as the handoff point for the future coordinator-weight receiver in Milestone 5
- perform no checkpoint loading, downloads, or remote execution inside the worker
- identify experts using `(layer_id, expert_id)`
- execute the requested expert on a provided hidden-state tensor
- return the resulting tensor

Acceptance criteria:

- a worker can host multiple experts
- requesting different expert IDs executes the correct expert
- requesting an unknown expert fails cleanly
- all tests run locally without networking

---

### Milestone 4 — Tensor Transport

Implement communication between two processes without integrating the model yet.

Requirements:

- establish communication between two processes
- send a PyTorch tensor from one process to another
- serialize and deserialize tensors
- preserve tensor shape and dtype
- return a tensor response
- measure:
  - serialization time
  - payload size
  - transmission time

Acceptance criteria:

- Process A can send a tensor to Process B
- Process B receives an equivalent tensor
- Process B can return a tensor to Process A
- received tensors preserve shape, dtype, and values within the expected tolerance
- transport code contains no model-specific logic

---

### Milestone 5 — Startup Expert Distribution and Worker Initialization

Use the local checkpoint loading from Milestone 2, the expert worker from
Milestone 3, and the transport layer from Milestone 4 to initialize worker nodes.
The coordinator sends every assigned remote expert on every system startup.

Requirements:

- accept an explicit list of expert assignments for each node during initialization
- load assigned experts from the coordinator's local checkpoint
- send every assigned remote expert's weights and the metadata needed to reconstruct it using the transport layer
- reconstruct and register the assigned experts in each receiving node's `ExpertWorker`
- confirm that all required workers are ready before inference begins
- fail explicitly on missing weights, incompatible metadata, failed transfers, or initialization timeouts
- keep startup weight distribution separate from inference requests while reusing the transport abstraction

Acceptance criteria:

- a receiving worker can reconstruct, register, and locally execute its assigned experts using weights sent by the coordinator
- execution on the receiving worker matches local execution with the same weights and inputs within floating-point tolerance
- restarting the system sends all assigned remote expert weights again, without relying on a persistent node cache
- receiving nodes obtain only their assigned experts and do not need to download the model from Hugging Face
- initialization reports readiness only after all required experts have been received and loaded
- automated initialization tests cover transfers between processes using small local fixtures without requiring Hugging Face access

Persistent node caching and incremental weight transfers are deferred. The
runtime placement interface remains in Milestone 7.

---

### Milestone 6 — Remote Expert Execution

Connect the transport layer to the expert worker.

Requirements:

- Node A sends:
  - `layer_id`
  - `expert_id`
  - hidden states

- Node B locates the requested expert
- Node B executes the expert using PyTorch
- Node B returns the expert output
- Node A reconstructs the returned tensor

Acceptance criteria:

- an expert can be invoked remotely
- the remote expert output is numerically equivalent to local execution within floating-point tolerance
- failures such as unknown expert IDs return a controlled error
- model logic remains separate from transport logic

---

### Milestone 7 — Static Expert Placement

Introduce an abstraction that determines where each expert is located.

Requirements:

- create a configurable mapping from `(layer_id, expert_id)` to a node
- support both local and remote expert locations
- do not hard-code node locations inside the model
- expose expert placement through a dedicated interface

Example:

```text
(layer 0, expert 0) -> node-a
(layer 0, expert 1) -> node-b
(layer 0, expert 2) -> node-a
```

Acceptance criteria:

- expert ownership can be changed through configuration
- changing expert placement does not require changes to model logic
- the system correctly decides whether to execute an expert locally or remotely

---

### Milestone 8 — MoE Routing and Dispatch

Implement the routing and dispatch logic required for a Mixture-of-Experts layer.

Requirements:

- compute router scores
- select the top-k experts for each token
- group tokens by selected expert
- execute local experts locally
- execute remote experts through the remote executor
- preserve router weights
- reassemble expert outputs into the correct token order

Acceptance criteria:

- tokens are dispatched to the experts selected by the router
- local and remote experts can participate in the same forward pass
- output ordering is correct
- weighted expert outputs are aggregated correctly
- tests cover multiple tokens and multiple selected experts

---

### Milestone 9 — End-to-End Two-Node Prototype

Integrate the previous components into a complete distributed MoE forward pass.

Target topology:

```text
Node A
├── shared model components
├── router
├── local experts
├── expert placement
└── remote expert executor
          │
          │ network
          ▼
Node B
├── expert worker
└── remote experts
```

Requirements:

- Node A runs the shared model components
- the router selects experts
- local experts execute on Node A
- remote expert inputs are sent to Node B
- Node B performs expert inference
- Node B returns expert outputs
- Node A aggregates the outputs
- Node A continues the forward pass

Acceptance criteria:

- a complete MoE layer can execute using experts distributed across two nodes
- distributed execution produces numerically equivalent results to an equivalent fully local execution within floating-point tolerance
- the model does not directly depend on the transport implementation

---

### Milestone 10 — Benchmarking and Instrumentation

Add measurements required to evaluate distributed expert inference.

Measure:

- serialization time
- deserialization time
- request payload size
- response payload size
- request transmission time
- expert computation time
- response transmission time
- total remote expert invocation time
- local expert invocation time

Where useful, also report:

- average latency
- p50 latency
- p95 latency
- p99 latency
- throughput
- bytes transferred per expert invocation

Acceptance criteria:

- a benchmark can compare local and remote expert execution
- individual latency components are reported separately
- benchmark results can be saved for later analysis
- measurement code does not alter the expert execution behavior

---

## V1 Completion

V1 is complete when Milestones 1 through 10 are implemented and tested.

The resulting prototype should demonstrate that an MoE expert can be stored on a different node, invoked transparently during inference, and evaluated in terms of correctness and communication performance.

Features listed under **Future Scope** are not required for V1.

## Requirements

- Python
- PyTorch

Transport measurements are defined in [Milestone 4](#milestone-4--tensor-transport),
and the full benchmarking requirements in [Milestone 10](#milestone-10--benchmarking-and-instrumentation).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the component overview,
[design principles](ARCHITECTURE.md#design-principles), and
[proposed project structure](ARCHITECTURE.md#project-structure).

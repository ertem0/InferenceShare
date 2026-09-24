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
Milestone 4, including the communication needed for startup transfers.

---

### Milestone 3 — Expert Worker

Implement a worker capable of hosting and executing multiple experts locally.

Requirements:

- create an `ExpertWorker`
- register multiple already constructed local PyTorch expert modules supplied by initialization code
- expose registration as the handoff point for the future coordinator-weight receiver in Milestone 4
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

### Milestone 4 — Coordinator Startup and Worker Initialization

Build coordinator initialization around the local checkpoint loading from
Milestone 2 and the expert worker from Milestone 3. Workers request assignments
using a configured memory budget. The coordinator sends every assigned expert
on every system startup, without relying on a persistent worker cache.

Coordinator requirements:

- maintain an inventory of every `(layer_id, expert_id)` in the model, including unallocated experts, assigned node, and allocation status (unallocated, loading, or ready)
- listen for worker initialization requests containing the worker's allocated memory budget
- assume all experts have the same weight-storage size in the selected loading dtype
- allocate the first unallocated experts in ascending `(layer_id, expert_id)` order, up to `min(unallocated_count, memory_budget_bytes // expert_size_bytes)`
- treat the reported budget as expert weight storage; the worker must leave memory outside this budget for loading overhead and execution
- reserve assignments before transferring weights so simultaneous initialization requests cannot assign the same expert to different nodes
- load assigned experts from the coordinator's local checkpoint and send their identities, reconstruction metadata, and weights
- send an explicit assignment list so each worker knows which experts it must load
- receive readiness or load-error reports identifying failed experts
- resend only failed experts, with at most three total attempts per expert, preserving successfully loaded experts
- after attempts are exhausted or an expert is rejected, release its assignment while retaining its inventory entry as unallocated
- confirm the reduced assignment with the worker before marking it ready; readiness requires successful loading of every expert in the final confirmed assignment
- monitor each node with heartbeats and a timeout
- on disconnect or health timeout, release all of that node's assignments, including reservations, and mark those experts unallocated

Worker requirements:

- start with the coordinator address and an allocated memory budget, then send an initialization request
- receive the assignment list, reconstruction metadata, and expert weights
- reconstruct fresh local modules and register them in `ExpertWorker`; keep checkpoint access and communication outside `ExpertWorker`
- report readiness when all assigned experts load successfully, or report a load error listing the failed experts
- retain successful experts during retries and handle repeated transfers without duplicate-registration failures
- accept and acknowledge a reduced final assignment after failed experts are released
- participate in health monitoring and, once ready, wait for execution instructions; remote execution is implemented in Milestone 6

Communication requirements:

- use TCP between processes on the same machine for initial tests
- listen on separate control and tensor ports; initialize and exchange heartbeats on the control connection
- after initialization confirmation, attach the worker's tensor connection using its node ID and session token, rejecting unknown or duplicate attachments
- mark a node ready only after its final expert assignment is loaded and its tensor connection is attached; apply a timeout to attachment
- keep identified connections in node sessions and unidentified sockets in a pending set; public snapshots exclude sockets and session tokens
- close both channels and release assignments if either channel fails; tensor execution messages remain deferred to later milestones
- frame messages as `[length][message type][metadata length][metadata][payload]`, with an explicit metadata boundary
- support CPU weight tensors with float32, float16, and bfloat16 dtypes
- keep model-specific reconstruction outside the communication layer
- transfer weights one-way from coordinator to worker, with readiness and error reports flowing back
- fail explicitly on missing weights, incompatible metadata, malformed messages, failed transfers, or initialization timeouts
- keep startup weight distribution separate from inference requests while reusing the communication abstraction

Acceptance criteria:

- allocation selects the first unallocated equal-sized experts that fit the reported budget, without duplicate ownership
- a receiving worker can reconstruct, register, and locally execute its assigned experts using coordinator-supplied weights
- execution on the receiving worker matches local execution with the same weights and inputs within floating-point tolerance
- retries target only failed experts and stop after three total attempts per expert
- successful experts remain loaded when other experts fail; failed assignments become unallocated and the reduced assignment is confirmed before readiness
- disconnected or unhealthy nodes release all their assignments
- restarting the system sends all assigned expert weights again; workers do not download the model from Hugging Face
- automated tests use small local fixtures and cover allocation, reconstruction, retries, reduced assignments, health monitoring, and transfers between local processes without Hugging Face access

Persistent node caching, incremental weight transfers, and automatic redistribution
after a disconnect are deferred. Released experts are eligible for subsequent
initialization requests. The runtime placement interface remains in Milestone 7;
this milestone owns only the startup allocation inventory. Milestone 5 builds on
this communication layer for standalone tensor exchanges and transport measurements.

---

### Milestone 5 — Tensor Transport

Build on Milestone 4's communication layer to support and measure standalone
tensor transfers between two processes on the same machine, without integrating
model inference yet.

Requirements:

- expose independent tensor send and receive operations, with request/response built on top
- support one-way tensor transfers without requiring a tensor response
- support a tensor echo request/response
- serialize and deserialize CPU tensors of arbitrary shape with float32, float16, or bfloat16 dtype
- preserve tensor shape, dtype, and values; strides and gradients need not be preserved
- fail explicitly on unsupported inputs, malformed messages, disconnects, or timeouts
- keep transport code independent of model-specific logic and reusable by startup initialization
- measure:
  - serialization and deserialization time
  - tensor payload size and total framed message size in bytes
  - socket-send duration: time spent handing the framed message to the socket, not confirmed delivery time
  - echo round-trip duration: from starting the request send until the complete response arrives, including server processing
  - total echo invocation duration: from before request serialization through response deserialization

Acceptance criteria:

- Process A can send a tensor to Process B independently of a response
- Process B receives an equivalent tensor and can return a tensor response for an echo exchange
- received tensors preserve shape, dtype, and values within the expected tolerance
- automated tests cover one-way and echo transfers between local processes with explicit readiness synchronization
- tests cover supported dtypes, arbitrary shapes, empty tensors, noncontiguous inputs, and controlled failures
- timing and byte-count measurements are available for transfers; round-trip and invocation timings apply to echo exchanges
- no synchronized clocks or true one-way network latency measurements are required
- startup expert distribution from Milestone 4 continues to work through the shared communication layer

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

Transport measurements are defined in [Milestone 5](#milestone-5--tensor-transport),
and the full benchmarking requirements in [Milestone 10](#milestone-10--benchmarking-and-instrumentation).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the component overview,
[design principles](ARCHITECTURE.md#design-principles), and
[proposed project structure](ARCHITECTURE.md#project-structure).

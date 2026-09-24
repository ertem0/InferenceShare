# Architecture

## Overview

The system separates model execution, expert placement, and communication.

```text
Model
  |
  v
Router
  |
  v
Expert Executor
  |
  +---- Local Expert Executor
  |
  +---- Remote Expert Executor
              |
              v
           Transport
              |
              v
         Expert Worker
```

## Design principles

- **Separate model logic from networking.** PyTorch model components should not
  depend directly on sockets, RPC implementations, or transport-specific details.
- **Use compatible execution interfaces.** Local and remote expert execution
  should expose compatible interfaces. Workers should be able to host multiple
  experts identified by `(layer_id, expert_id)`.
- **Separate routing from placement.** The router selects experts for each token;
  a dedicated, configurable placement interface determines where those experts
  reside. Do not embed node addresses or worker identities in model code.
- **Keep transport replaceable.** Isolate networking behind an interface so
  alternative transports can be evaluated without rewriting model logic.
- **Keep V1 simple.** Future requirements should inform component boundaries,
  but should not introduce abstractions or complexity without a current need.

## Project structure

The proposed layout separates the main responsibilities below. It may evolve as
implementation progresses.

```text
src/
    SharedInference/
        __init__.py
        model/
        routing/
        experts/
        networking/
        runtime/
        benchmarks/

tests/
    unit/
    integration/
```

`runtime/` owns startup allocation, coordinator and worker sessions, and the CLI
entry point (`python -m SharedInference.runtime.cli`). It composes model adapters,
local expert execution, and networking. `networking/` provides message framing
and weight serialization. Session code currently manages TCP sockets directly;
this layout does not introduce an additional transport abstraction.

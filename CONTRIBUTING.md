# Contributing

This project is a research prototype for distributed Mixture-of-Experts inference.

Contributions should prioritize correctness, modularity, reproducibility, and clear experimental behavior over premature optimization.

## Before making changes

Before implementing a feature:

1. Read `PROJECT.md`.
2. Read `ARCHITECTURE.md`.
3. Inspect the existing code before creating new abstractions.
4. Confirm which milestone or issue the change belongs to.
5. Keep the implementation limited to the current task.

Do not implement features listed under Future Scope unless they are explicitly part of the current task.

## Development setup

The project uses:

- Python
- `uv` for dependency and environment management
- PyTorch
- Ruff for linting and formatting
- pytest for testing

Synchronize the project environment with:

```bash
uv sync
```

Run commands through `uv` rather than relying on a manually activated virtual environment.

Example:

```bash
uv run pytest
```

## Development commands

### Run tests

Run the complete test suite:

```bash
uv run pytest
```

Run tests with more detailed output:

```bash
uv run pytest -v
```

Run only unit tests:

```bash
uv run pytest tests/unit
```

Run only integration tests:

```bash
uv run pytest tests/integration
```

Run a specific test file:

```bash
uv run pytest tests/unit/test_expert_execution.py
```

### Lint

Check the codebase with Ruff:

```bash
uv run ruff check .
```

Automatically fix safe linting issues:

```bash
uv run ruff check . --fix
```

### Format

Format the codebase:

```bash
uv run ruff format .
```

Check formatting without modifying files:

```bash
uv run ruff format --check .
```

## Before committing

Before committing changes, run:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

All commands should complete successfully.

Do not commit code with known failing tests unless the failure is intentionally part of an isolated work-in-progress branch.

## Architecture

Follow the [design principles](ARCHITECTURE.md#design-principles) and
[proposed project structure](ARCHITECTURE.md#project-structure) in `ARCHITECTURE.md`.

Do not introduce new top-level modules or architectural layers without a clear reason.

## Testing guidelines

Every new behavior should have corresponding tests.

### Unit tests

Unit tests should:

- be fast
- avoid real network communication when possible
- test one component or responsibility
- be deterministic
- avoid relying on external services

Examples include:

- expert execution
- expert registration
- expert placement
- serialization
- routing logic
- output aggregation

### Integration tests

Integration tests should verify interactions between components.

Examples include:

- communication between two processes
- tensor round trips
- remote expert execution
- local versus remote numerical equivalence
- complete distributed MoE layer execution

### Numerical correctness

When comparing PyTorch outputs, use appropriate floating-point tolerances rather than exact equality.

For example:

```python
torch.testing.assert_close(
    actual,
    expected,
    rtol=1e-5,
    atol=1e-6,
)
```

Use tolerances appropriate for the datatype and operation being tested.

### Distributed behavior

Tests involving multiple processes or nodes should avoid relying unnecessarily on timing.

Prefer explicit synchronization and deterministic request identifiers.

Failures should be reproducible whenever possible.

## Error handling

Distributed components should fail explicitly rather than silently.

Examples of conditions that should produce controlled errors include:

- unknown expert IDs
- unknown layer IDs
- malformed requests
- incompatible tensor metadata
- unavailable workers
- connection failure
- request timeout

Avoid broad exception handling

"""Run a local startup coordinator or a worker waiting for execution."""

import argparse
import threading

import torch

from SharedInference.model.olmoe_initialization import (
    OlmoeExpertSource,
    reconstruct_olmoe_expert,
)

from .coordinator_server import Coordinator
from .worker_client import WorkerClient


def main():
    invocation = "uv run python -m SharedInference.runtime.cli"
    parser = argparse.ArgumentParser(
        description="Initialize distributed MoE expert workers over separate control and tensor TCP connections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Start in separate terminals:\n"
            f"  {invocation} coordinator --checkpoint /path/to/checkpoint\n"
            f"  {invocation} worker --memory-bytes 268435456\n\n"
            f"Command help: {invocation} <command> --help\n"
            "Press Ctrl-C to stop either process. Remote inference is not implemented yet."
        ),
    )
    commands = parser.add_subparsers(dest="role", required=True, title="commands")
    coordinator = commands.add_parser(
        "coordinator",
        help="Assign experts from a local checkpoint and initialize workers.",
        description=(
            "Load assigned OLMoE experts from a local checkpoint and send them to workers. "
            "Listen on separate control and tensor ports, then monitor worker health."
        ),
        epilog="The checkpoint must already be downloaded. Press Ctrl-C to stop.",
    )
    coordinator.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Local OLMoE checkpoint directory containing config, index, and Safetensors shards.",
    )
    coordinator.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="CPU dtype used to load, transfer, and account for expert weights (default: %(default)s).",
    )
    coordinator.add_argument(
        "--tensor-port",
        type=int,
        default=5001,
        help="TCP listening port reserved for tensor traffic; must differ from the control port (default: %(default)s).",
    )
    worker = commands.add_parser(
        "worker",
        help="Connect to a coordinator and host assigned experts.",
        description=(
            "Request experts using a weight-memory budget, reconstruct them locally, "
            "and attach the tensor connection before becoming ready. "
            "The coordinator supplies the tensor port; no local checkpoint is needed."
        ),
        epilog="After initialization, answer heartbeats and wait. Press Ctrl-C to stop.",
    )
    worker.add_argument(
        "--memory-bytes",
        type=int,
        required=True,
        metavar="BYTES",
        help=(
            "Nonnegative expert weight-storage budget in bytes "
            "(268435456 = 256 MiB). Leave additional RAM for loading and execution."
        ),
    )
    coordinator.add_argument(
        "--host",
        default="127.0.0.1",
        help="Local IPv4 address to bind both listeners to (default: %(default)s).",
    )
    worker.add_argument(
        "--host",
        default="127.0.0.1",
        help="Coordinator hostname or IPv4 address to connect to (default: %(default)s).",
    )
    for command in (coordinator, worker):
        command.add_argument(
            "--control-port",
            type=int,
            default=5000,
            help="Coordinator TCP port for initialization and heartbeats (default: %(default)s).",
        )
    args = parser.parse_args()
    control_address = (args.host, args.control_port)
    try:
        if args.role == "coordinator":
            source = OlmoeExpertSource(
                args.checkpoint, dtype=getattr(torch, args.dtype)
            )
            with Coordinator(
                source.expert_ids,
                source.expert_size_bytes,
                source.prepare,
                control_address=control_address,
                tensor_address=(args.host, args.tensor_port),
            ) as server:
                print(
                    f"Coordinator control at {server.control_address}, tensors at {server.tensor_address}; {source.expert_size_bytes} bytes per expert",
                    flush=True,
                )
                threading.Event().wait()
        else:
            with WorkerClient(
                control_address, args.memory_bytes, reconstruct_olmoe_expert
            ) as client:
                print(
                    f"Worker {client.node_id} ready: {client.assignments}", flush=True
                )
                client.wait_until_disconnected()
                if client.error:
                    raise RuntimeError(client.error)
    except KeyboardInterrupt:
        pass  # Context managers close connections and release assignments.


if __name__ == "__main__":
    main()

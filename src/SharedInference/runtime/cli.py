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
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="role", required=True)
    coordinator = commands.add_parser("coordinator")
    coordinator.add_argument("--checkpoint", required=True)
    coordinator.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    coordinator.add_argument("--tensor-port", type=int, default=5001)
    worker = commands.add_parser("worker")
    worker.add_argument("--memory-bytes", type=int, required=True)
    for command in (coordinator, worker):
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--control-port", type=int, default=5000)
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

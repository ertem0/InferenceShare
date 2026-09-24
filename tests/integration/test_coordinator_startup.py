import multiprocessing
import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from SharedInference.model.olmoe import load_olmoe_expert
from SharedInference.model.olmoe_initialization import (
    OlmoeExpertSource,
    reconstruct_olmoe_expert,
)
from SharedInference.networking.protocol import Message, expect_message, send_message
from SharedInference.runtime import Coordinator, WorkerClient
from SharedInference.runtime.worker_client import ExpertRejected


def coordinator(source, **kwargs):
    return Coordinator(
        source.expert_ids,
        source.expert_size_bytes,
        source.prepare,
        heartbeat_interval=0.02,
        health_timeout=0.2,
        **kwargs,
    )


def test_first_unallocated_and_disconnect(olmoe_checkpoint):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    with coordinator(source) as server:
        with WorkerClient(
            server.control_address,
            source.expert_size_bytes * 2 + 1,
            reconstruct_olmoe_expert,
        ) as first:
            assert first.assignments == ((0, 0), (0, 1))
            first_id = first.node_id
            with WorkerClient(
                server.control_address,
                source.expert_size_bytes,
                reconstruct_olmoe_expert,
            ) as second:
                assert second.assignments == ((1, 0),)
                assert server.inventory[(1, 1)]["status"] == "unallocated"
        assert server.wait_for_node(first_id, "disconnected")
        with WorkerClient(
            server.control_address, source.expert_size_bytes, reconstruct_olmoe_expert
        ) as replacement:
            assert replacement.assignments == ((0, 0),)


@pytest.mark.parametrize("budget", [0, 71])
def test_budget_too_small_is_empty_ready_assignment(olmoe_checkpoint, budget):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    with (
        coordinator(source) as server,
        WorkerClient(
            server.control_address, budget, reconstruct_olmoe_expert
        ) as client,
    ):
        assert client.assignments == ()
        assert client.ready.is_set()
        assert all(entry["node_id"] is None for entry in server.inventory.values())


def test_concurrent_reservations(olmoe_checkpoint):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    with coordinator(source) as server:
        clients = [
            WorkerClient(
                server.control_address,
                source.expert_size_bytes * 3,
                reconstruct_olmoe_expert,
            )
            for _ in range(2)
        ]
        try:
            with ThreadPoolExecutor(2) as pool:
                list(pool.map(lambda client: client.initialize(), clients))
            assignments = [item for client in clients for item in client.assignments]
            assert len(assignments) == len(set(assignments)) == 4
        finally:
            for client in clients:
                client.close()


@pytest.mark.parametrize("reject", [False, True])
def test_failed_experts_released_and_successes_retained(olmoe_checkpoint, reject):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    calls = {}

    def reconstruct(layer, expert, metadata, weights):
        item = (layer, expert)
        calls[item] = calls.get(item, 0) + 1
        if item == (0, 1):
            if reject:
                raise ExpertRejected("Cannot host this expert")
            raise RuntimeError("Simulated load failure")
        return reconstruct_olmoe_expert(layer, expert, metadata, weights)

    with (
        coordinator(source) as server,
        WorkerClient(
            server.control_address, source.expert_size_bytes * 3, reconstruct
        ) as client,
    ):
        assert client.assignments == ((0, 0), (1, 0))
        assert calls == {(0, 0): 1, (0, 1): 1 if reject else 3, (1, 0): 1}
        assert server.inventory[(0, 1)] == {
            "node_id": None,
            "status": "unallocated",
        }
        assert server.nodes[client.node_id]["status"] == "ready"
        for item in client.assignments:
            expected = load_olmoe_expert(olmoe_checkpoint[0], *item).module
            inputs = torch.ones(2, 2)
            with torch.inference_mode():
                torch.testing.assert_close(
                    client.worker.execute(*item, inputs), expected(inputs)
                )


def test_transient_failure_recovers(olmoe_checkpoint):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    attempts = []

    def reconstruct(*args):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("Temporary load failure")
        return reconstruct_olmoe_expert(*args)

    with (
        coordinator(source) as server,
        WorkerClient(
            server.control_address, source.expert_size_bytes, reconstruct
        ) as client,
    ):
        assert client.assignments == ((0, 0),)
        assert len(attempts) == 3


def test_health_timeout_releases_assignments(olmoe_checkpoint):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    with (
        coordinator(source) as server,
        socket.create_connection(server.control_address, timeout=3) as connection,
    ):
        send_message(
            connection,
            Message.INITIALIZE,
            {"memory_budget_bytes": source.expert_size_bytes},
        )
        assignment, _ = expect_message(connection, Message.ASSIGN)
        weights, _ = expect_message(connection, Message.WEIGHTS)
        send_message(
            connection,
            Message.RESULT,
            {"expert": weights["expert"], "status": "loaded"},
        )
        final, _ = expect_message(connection, Message.FINALIZE)
        send_message(connection, Message.READY, final)
        confirmed, _ = expect_message(connection, Message.CONFIRMED)
        tensor = socket.create_connection(server.tensor_address, timeout=3)
        try:
            send_message(
                tensor,
                Message.ATTACH_TENSOR,
                {
                    "node_id": confirmed["node_id"],
                    "session_token": confirmed["session_token"],
                },
            )
            expect_message(tensor, Message.TENSOR_ATTACHED)
            expect_message(connection, Message.PING)
            assert server.wait_for_node(assignment["node_id"], "disconnected")
        finally:
            tensor.close()
        # Deliberately never answer the heartbeat. No timing-based sleep.
        assert server.wait_for_node(assignment["node_id"], "disconnected")
        assert "TimeoutError" in server.nodes[assignment["node_id"]]["error"]
        assert all(entry["node_id"] is None for entry in server.inventory.values())


def test_missing_checkpoint_weights_fail_explicitly(olmoe_checkpoint):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    (olmoe_checkpoint[0] / "shard-0.safetensors").unlink()
    with coordinator(source) as server:
        client = WorkerClient(
            server.control_address, source.expert_size_bytes, reconstruct_olmoe_expert
        )
        with pytest.raises(ValueError, match="FileNotFoundError"):
            client.initialize()
        assert server.wait_for_node(client.node_id, "disconnected")
        assert all(entry["node_id"] is None for entry in server.inventory.values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_two_process_initialization(olmoe_checkpoint, dtype):
    # Child receives only the control_address and budget, never a checkpoint path.
    source = OlmoeExpertSource(olmoe_checkpoint[0], dtype=dtype)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    with coordinator(source) as server:
        process = ctx.Process(
            target=_worker_process_typed,
            args=(server.control_address, source.expert_size_bytes, child, dtype),
        )
        process.start()
        child.close()
        try:
            assert parent.poll(20), "Worker did not report initialization"
            node_id, assignments, values = parent.recv()
            assert assignments == ((0, 0),)
            assert server.wait_for_node(node_id, "ready")
            expected = load_olmoe_expert(olmoe_checkpoint[0], 0, 0, dtype=dtype)
            with torch.inference_mode():
                torch.testing.assert_close(
                    torch.tensor(values, dtype=dtype),
                    expected.module(torch.ones(2, 2, dtype=dtype)),
                )
            parent.send("stop")
            process.join(10)
            assert process.exitcode == 0
            assert server.wait_for_node(node_id, "disconnected")
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            parent.close()


def _worker_process_typed(control_address, budget, pipe, dtype):
    with WorkerClient(control_address, budget, reconstruct_olmoe_expert) as client:
        result = client.worker.execute(0, 0, torch.ones(2, 2, dtype=dtype))
        pipe.send((client.node_id, client.assignments, result.tolist()))
        assert pipe.recv() == "stop"
    pipe.close()


def test_disconnect_during_initialization_releases_reservation(olmoe_checkpoint):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    with coordinator(source) as server:
        with socket.create_connection(server.control_address, timeout=3) as connection:
            send_message(
                connection,
                Message.INITIALIZE,
                {"memory_budget_bytes": source.expert_size_bytes},
            )
            assignment, _ = expect_message(connection, Message.ASSIGN)
            expect_message(connection, Message.WEIGHTS)
        assert server.wait_for_node(assignment["node_id"], "disconnected")
        assert server.inventory[(0, 0)] == {"node_id": None, "status": "unallocated"}


def test_incorrect_final_confirmation_never_becomes_ready(olmoe_checkpoint):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    with (
        coordinator(source) as server,
        socket.create_connection(server.control_address, timeout=3) as connection,
    ):
        send_message(
            connection,
            Message.INITIALIZE,
            {"memory_budget_bytes": source.expert_size_bytes},
        )
        assignment, _ = expect_message(connection, Message.ASSIGN)
        weights, _ = expect_message(connection, Message.WEIGHTS)
        send_message(
            connection,
            Message.RESULT,
            {"expert": weights["expert"], "status": "loaded"},
        )
        expect_message(connection, Message.FINALIZE)
        send_message(connection, Message.READY, {"experts": []})
        assert server.wait_for_node(assignment["node_id"], "disconnected")
        assert "final assignment" in server.nodes[assignment["node_id"]]["error"]


def test_initialization_timeout_releases_reservation(olmoe_checkpoint):
    source = OlmoeExpertSource(olmoe_checkpoint[0])
    with (
        coordinator(source, initialization_timeout=0.1) as server,
        socket.create_connection(server.control_address, timeout=3) as connection,
    ):
        send_message(
            connection,
            Message.INITIALIZE,
            {"memory_budget_bytes": source.expert_size_bytes},
        )
        assignment, _ = expect_message(connection, Message.ASSIGN)
        expect_message(connection, Message.WEIGHTS)
        assert server.wait_for_node(assignment["node_id"], "disconnected")
        assert "TimeoutError" in server.nodes[assignment["node_id"]]["error"]
        assert server.inventory[(0, 0)]["node_id"] is None


def test_repeated_transfer_does_not_register_twice(olmoe_checkpoint):
    import threading

    from SharedInference.networking.protocol import encode_weights

    source = OlmoeExpertSource(olmoe_checkpoint[0])
    metadata, weights = source.prepare(0, 0)
    payload = encode_weights(weights)
    calls = []
    finish = threading.Event()

    def reconstruct(*args):
        calls.append(1)
        return reconstruct_olmoe_expert(*args)

    with socket.socket() as listener, socket.socket() as tensor_listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        tensor_listener.bind(("127.0.0.1", 0))
        tensor_listener.listen()
        tensor_listener.settimeout(5)

        def serve():
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(5)
                expect_message(connection, Message.INITIALIZE)
                send_message(
                    connection,
                    Message.ASSIGN,
                    {
                        "node_id": "test-node",
                        "experts": [[0, 0]],
                        "expert_size_bytes": source.expert_size_bytes,
                        "heartbeat_interval": 1,
                        "health_timeout": 5,
                    },
                )
                for attempt in (1, 2):
                    send_message(
                        connection,
                        Message.WEIGHTS,
                        {"expert": [0, 0], "model": metadata, "attempt": attempt},
                        payload,
                    )
                    report, _ = expect_message(connection, Message.RESULT)
                    assert report["status"] == "loaded"
                send_message(connection, Message.FINALIZE, {"experts": [[0, 0]]})
                ready, _ = expect_message(connection, Message.READY)
                assert ready["experts"] == [[0, 0]]
                send_message(
                    connection,
                    Message.CONFIRMED,
                    {
                        "node_id": "test-node",
                        "session_token": "test-token",
                        "tensor_port": tensor_listener.getsockname()[1],
                    },
                )
                tensor, _ = tensor_listener.accept()
                with tensor:
                    tensor.settimeout(5)
                    attachment, _ = expect_message(tensor, Message.ATTACH_TENSOR)
                    assert attachment == {
                        "node_id": "test-node",
                        "session_token": "test-token",
                    }
                    send_message(
                        tensor, Message.TENSOR_ATTACHED, {"node_id": "test-node"}
                    )
                    send_message(connection, Message.PING, {"sequence": 0})
                    expect_message(connection, Message.PONG)
                    assert finish.wait(5)

        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(serve)
            try:
                with WorkerClient(
                    listener.getsockname(), source.expert_size_bytes, reconstruct
                ) as client:
                    assert client.assignments == ((0, 0),)
                    assert calls == [1]
                    finish.set()
                    assert client.wait_until_disconnected(5)
                    assert not client.ready.is_set()
                future.result(timeout=5)
            finally:
                finish.set()

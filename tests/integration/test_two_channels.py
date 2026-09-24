import socket

import pytest

from SharedInference.model.olmoe_initialization import (
    OlmoeExpertSource,
    reconstruct_olmoe_expert,
)
from SharedInference.networking.protocol import (
    Message,
    expect_message,
    receive_message,
    send_message,
)
from SharedInference.runtime import Coordinator, WorkerClient


def server_for(checkpoint, **kwargs):
    source = OlmoeExpertSource(checkpoint)
    return Coordinator(
        source.expert_ids,
        source.expert_size_bytes,
        source.prepare,
        heartbeat_interval=0.02,
        health_timeout=0.5,
        **kwargs,
    )


def initialize_control(server, connection):
    send_message(
        connection,
        Message.INITIALIZE,
        {"memory_budget_bytes": server.expert_size_bytes},
    )
    expect_message(connection, Message.ASSIGN)
    weights, _ = expect_message(connection, Message.WEIGHTS)
    send_message(
        connection, Message.RESULT, {"expert": weights["expert"], "status": "loaded"}
    )
    final, _ = expect_message(connection, Message.FINALIZE)
    send_message(connection, Message.READY, final)
    confirmed, _ = expect_message(connection, Message.CONFIRMED)
    return confirmed


def attach(server, connection, confirmed):
    send_message(
        connection,
        Message.ATTACH_TENSOR,
        {
            "node_id": confirmed["node_id"],
            "session_token": confirmed["session_token"],
        },
    )
    return receive_message(connection)


def test_ready_requires_second_port_and_snapshot_excludes_connections(olmoe_checkpoint):
    with (
        server_for(olmoe_checkpoint[0]) as server,
        socket.create_connection(server.control_address, timeout=3) as control,
    ):
        confirmed = initialize_control(server, control)
        node_id = confirmed["node_id"]
        assert server.control_address[1] != server.tensor_address[1]
        assert server.nodes[node_id]["status"] == "initialized"
        assert server.inventory[(0, 0)]["status"] == "loading"
        with socket.create_connection(server.tensor_address, timeout=3) as tensor:
            assert attach(server, tensor, confirmed)[0] == Message.TENSOR_ATTACHED
            assert server.wait_for_node(node_id, "ready")
            node = server.nodes[node_id]
            assert set(node) == {"status", "error", "experts", "memory_budget_bytes"}
            node["experts"].clear()
            assert server.nodes[node_id]["experts"] == [(0, 0)]
            # An idle tensor connection must not block control heartbeats.
            for _ in range(3):
                ping, _ = expect_message(control, Message.PING)
                send_message(control, Message.PONG, ping)
        assert server.wait_for_node(node_id, "disconnected")
        assert server.inventory[(0, 0)]["node_id"] is None


def test_missing_tensor_attachment_times_out(olmoe_checkpoint):
    with (
        server_for(olmoe_checkpoint[0], attachment_timeout=0.1) as server,
        socket.create_connection(server.control_address, timeout=3) as control,
    ):
        confirmed = initialize_control(server, control)
        assert server.wait_for_node(confirmed["node_id"], "disconnected")
        assert "attachment timed out" in server.nodes[confirmed["node_id"]]["error"]
        assert server.inventory[(0, 0)]["node_id"] is None


def test_invalid_and_duplicate_attachments_do_not_disconnect_owner(olmoe_checkpoint):
    with (
        server_for(olmoe_checkpoint[0]) as server,
        socket.create_connection(server.control_address, timeout=3) as control,
    ):
        confirmed = initialize_control(server, control)
        for changed in ({"session_token": "wrong"}, {"node_id": "unknown"}):
            with socket.create_connection(server.tensor_address, timeout=3) as invalid:
                assert attach(server, invalid, confirmed | changed)[0] == Message.ERROR
            assert server.nodes[confirmed["node_id"]]["status"] == "initialized"
        with socket.create_connection(server.tensor_address, timeout=3) as tensor:
            assert attach(server, tensor, confirmed)[0] == Message.TENSOR_ATTACHED
            with socket.create_connection(
                server.tensor_address, timeout=3
            ) as duplicate:
                assert attach(server, duplicate, confirmed)[0] == Message.ERROR
            assert server.nodes[confirmed["node_id"]]["status"] == "ready"


@pytest.mark.parametrize("channel", ["_control_connection", "_tensor_connection"])
def test_either_channel_failure_clears_worker_and_inventory(olmoe_checkpoint, channel):
    with (
        server_for(olmoe_checkpoint[0]) as server,
        WorkerClient(
            server.control_address, server.expert_size_bytes, reconstruct_olmoe_expert
        ) as worker,
    ):
        node_id = worker.node_id
        getattr(worker, channel).shutdown(socket.SHUT_RDWR)
        assert worker.wait_until_disconnected(3)
        assert not worker.ready.is_set()
        assert server.wait_for_node(node_id, "disconnected")
        assert all(entry["node_id"] is None for entry in server.inventory.values())


def test_shutdown_closes_pending_connections_on_both_ports(olmoe_checkpoint):
    server = server_for(olmoe_checkpoint[0]).start()
    try:
        with (
            socket.create_connection(server.control_address, timeout=3) as control,
            socket.create_connection(server.tensor_address, timeout=3) as tensor,
        ):
            server.close()
            for connection in (control, tensor):
                # Closing an accepted socket can send ERROR before EOF; a queued
                # unaccepted connection can instead be reset by the OS.
                try:
                    while connection.recv(4096):
                        pass
                except ConnectionResetError:
                    pass
    finally:
        server.close()

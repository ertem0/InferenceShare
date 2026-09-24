import json
import socket
import struct
import threading

import pytest
import torch

from SharedInference.networking.protocol import (
    MAX_MESSAGE_BYTES,
    Message,
    ProtocolError,
    decode_weights,
    encode_weights,
    receive_message,
    send_message,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_weight_serialization(dtype):
    weights = {"weight": torch.arange(12, dtype=dtype).reshape(3, 4).T}
    received = decode_weights(encode_weights(weights))
    assert received["weight"].dtype == dtype
    torch.testing.assert_close(received["weight"], weights["weight"])


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"weight": torch.ones(2, dtype=torch.int64)},
        {"weight": torch.ones(2, device="meta")},
    ],
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        encode_weights(weights)


def test_fragmented_frame():
    metadata = json.dumps({"expert": [1, 2]}).encode()
    payload = b"weights"
    frame = (
        struct.pack(
            "!QBI", 5 + len(metadata) + len(payload), Message.WEIGHTS, len(metadata)
        )
        + metadata
        + payload
    )
    left, right = socket.socketpair()
    with left, right:
        thread = threading.Thread(
            target=lambda: [left.sendall(bytes([b])) for b in frame]
        )
        thread.start()
        kind, data, actual = receive_message(right)
        thread.join()
    assert kind == Message.WEIGHTS
    assert data == {"expert": [1, 2]}
    assert actual == payload


@pytest.mark.parametrize(
    "frame",
    [
        struct.pack("!Q", MAX_MESSAGE_BYTES + 1),
        struct.pack("!QBI", 5, 1, 1),
        struct.pack("!QBI", 7, 255, 2) + b"{}",
        struct.pack("!QBI", 7, 1, 2) + b"[]",
    ],
)
def test_malformed_frames(frame):
    left, right = socket.socketpair()
    with left, right:
        left.sendall(frame)
        with pytest.raises(ProtocolError):
            receive_message(right)


def test_truncated_frame():
    left, right = socket.socketpair()
    with left, right:
        left.sendall(b"\x00")
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(ConnectionError):
            receive_message(right)


def test_control_round_trip():
    left, right = socket.socketpair()
    with left, right:
        send_message(left, Message.INITIALIZE, {"memory_budget_bytes": 123})
        assert receive_message(right) == (
            Message.INITIALIZE,
            {"memory_budget_bytes": 123},
            b"",
        )

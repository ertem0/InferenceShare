"""Bounded TCP frames and CPU weight serialization for startup messages."""

import json
import socket
import struct
from enum import IntEnum

import torch
from safetensors.torch import load, save

MAX_MESSAGE_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class ProtocolError(ValueError):
    """A peer supplied an invalid startup message."""


class Message(IntEnum):
    INITIALIZE = 1
    ASSIGN = 2
    WEIGHTS = 3
    RESULT = 4
    FINALIZE = 5
    READY = 6
    CONFIRMED = 7
    PING = 8
    PONG = 9
    ERROR = 10
    ATTACH_TENSOR = 11
    TENSOR_ATTACHED = 12


def _read_exact(connection: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = connection.recv(min(length - len(data), 1024 * 1024))
        if not chunk:
            raise ConnectionError("Connection closed during message reception")
        data.extend(chunk)
    return bytes(data)


def send_message(connection, kind, metadata=None, payload=b""):
    """Send [uint64 body length][uint8 type][uint32 metadata length][JSON][bytes]."""
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ProtocolError("Metadata must be an object")
    encoded = json.dumps(metadata, allow_nan=False).encode("utf-8")
    size = 5 + len(encoded) + len(payload)
    if len(encoded) > MAX_METADATA_BYTES or size > MAX_MESSAGE_BYTES:
        raise ProtocolError("Message exceeds size limit")
    connection.sendall(struct.pack("!QBI", size, Message(kind), len(encoded)))
    connection.sendall(encoded)
    if payload:
        connection.sendall(payload)


def receive_message(connection):
    size = struct.unpack("!Q", _read_exact(connection, 8))[0]
    if not 5 <= size <= MAX_MESSAGE_BYTES:
        raise ProtocolError("Invalid message length")
    kind, metadata_size = struct.unpack("!BI", _read_exact(connection, 5))
    if metadata_size > MAX_METADATA_BYTES or metadata_size > size - 5:
        raise ProtocolError("Invalid metadata length")
    try:
        kind = Message(kind)
        metadata = json.loads(_read_exact(connection, metadata_size))
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError("Invalid message type or JSON metadata") from exc
    if not isinstance(metadata, dict):
        raise ProtocolError("Metadata must be an object")
    payload = _read_exact(connection, size - 5 - metadata_size)
    return kind, metadata, payload


def expect_message(connection, expected):
    kind, metadata, payload = receive_message(connection)
    if kind == Message.ERROR:
        raise ProtocolError(f"Peer error: {metadata.get('error', 'unspecified')}")
    if kind != expected:
        raise ProtocolError(f"Expected {expected.name}, received {kind.name}")
    if expected != Message.WEIGHTS and payload:
        raise ProtocolError("Unexpected payload on control message")
    return metadata, payload


def validate_weights(weights):
    if not weights:
        raise ValueError("Expert weights must not be empty")
    for tensor in weights.values():
        if (
            tensor.device.type != "cpu"
            or tensor.dtype not in DTYPES
            or tensor.layout != torch.strided
        ):
            raise ValueError(
                "Weights must be dense CPU float32, float16, or bfloat16 tensors"
            )
    return sum(t.numel() * t.element_size() for t in weights.values())


def encode_weights(weights):
    validate_weights(weights)
    return save(
        {name: tensor.detach().contiguous() for name, tensor in weights.items()}
    )


def decode_weights(payload):
    weights = load(payload)
    validate_weights(weights)
    return weights


def identity(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(part) is not int or part < 0 for part in value)
    ):
        raise ProtocolError("Expected nonnegative integer [layer_id, expert_id]")
    return tuple(value)


def identities(value):
    if not isinstance(value, list):
        raise ProtocolError("Expected an expert identity list")
    result = [identity(item) for item in value]
    if len(set(result)) != len(result):
        raise ProtocolError("Duplicate expert identities")
    return result

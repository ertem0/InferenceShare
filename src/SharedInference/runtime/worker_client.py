"""Startup receiver around a local ExpertWorker; no checkpoint access."""

import math
import socket
import threading

from safetensors import SafetensorError

from SharedInference.experts import ExpertWorker, IdentifiedExpert
from SharedInference.networking.protocol import (
    Message,
    ProtocolError,
    decode_weights,
    expect_message,
    identities,
    identity,
    receive_message,
    send_message,
    validate_weights,
)


class ExpertRejected(ValueError):
    """A reconstruction adapter rejects an expert without requesting retries."""


class WorkerClient:
    """Initialize a worker, then respond to health probes while awaiting execution.

    reconstruct(layer_id, expert_id, metadata, weights) returns IdentifiedExpert.
    Instances represent one two-channel session and cannot reconnect. close() releases
    local experts. A failed session also discards its local registrations.
    """

    def __init__(
        self, control_address, memory_budget_bytes, reconstruct, *, timeout=30.0
    ):
        if type(memory_budget_bytes) is not int or memory_budget_bytes < 0:
            raise ValueError("Memory budget must be a nonnegative integer")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite")
        self.control_address = control_address
        self.memory_budget_bytes = memory_budget_bytes
        self.reconstruct = reconstruct
        self.timeout = timeout
        self.worker = ExpertWorker()
        self.node_id = None
        self.assignments = ()
        self.ready = threading.Event()
        self.error = None
        self._control_connection = None
        self._thread = None
        self._tensor_connection = None
        self._tensor_thread = None
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._started = False

    def initialize(self):
        if self._started:
            raise RuntimeError("Worker client has already started")
        self._started = True
        try:
            self._control_connection = socket.create_connection(
                self.control_address, timeout=self.timeout
            )
            connection = self._control_connection
            send_message(
                connection,
                Message.INITIALIZE,
                {"memory_budget_bytes": self.memory_budget_bytes},
            )
            assignment, _ = expect_message(connection, Message.ASSIGN)
            assigned = identities(assignment.get("experts"))
            size = assignment.get("expert_size_bytes")
            if (
                type(size) is not int
                or size <= 0
                or len(assigned) * size > self.memory_budget_bytes
            ):
                raise ProtocolError(
                    "Assignment exceeds memory budget or has invalid expert size"
                )
            self.node_id = assignment.get("node_id")
            if not isinstance(self.node_id, str) or not self.node_id:
                raise ProtocolError("Missing node identity")
            interval = assignment.get("heartbeat_interval")
            health_timeout = assignment.get("health_timeout")
            for value in (interval, health_timeout):
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ProtocolError("Invalid heartbeat timing")
            loaded = set()
            attempts = {}
            while True:
                kind, metadata, payload = receive_message(connection)
                if kind == Message.ERROR:
                    raise ProtocolError(f"Coordinator error: {metadata.get('error')}")
                if kind == Message.FINALIZE:
                    final = identities(metadata.get("experts"))
                    if payload or set(final) != loaded:
                        raise ProtocolError(
                            "Final assignment must match successfully loaded experts"
                        )
                    send_message(connection, Message.READY, {"experts": final})
                    confirmed, _ = expect_message(connection, Message.CONFIRMED)
                    if confirmed.get("node_id") != self.node_id:
                        raise ProtocolError("Confirmation identifies the wrong node")
                    tensor_port = confirmed.get("tensor_port")
                    token = confirmed.get("session_token")
                    if (
                        type(tensor_port) is not int
                        or not 0 < tensor_port < 65536
                        or not isinstance(token, str)
                        or not token
                    ):
                        raise ProtocolError("Invalid tensor connection metadata")
                    self._tensor_connection = socket.create_connection(
                        (self.control_address[0], tensor_port), timeout=self.timeout
                    )
                    send_message(
                        self._tensor_connection,
                        Message.ATTACH_TENSOR,
                        {
                            "node_id": self.node_id,
                            "session_token": token,
                        },
                    )
                    attached, _ = expect_message(
                        self._tensor_connection, Message.TENSOR_ATTACHED
                    )
                    if attached.get("node_id") != self.node_id:
                        raise ProtocolError(
                            "Tensor attachment identifies the wrong node"
                        )
                    self._tensor_connection.settimeout(None)
                    self.assignments = tuple(final)
                    self.ready.set()
                    connection.settimeout(interval + health_timeout + self.timeout)
                    self._thread = threading.Thread(target=self._monitor, daemon=True)
                    self._tensor_thread = threading.Thread(
                        target=self._monitor_tensor, daemon=True
                    )
                    self._thread.start()
                    self._tensor_thread.start()
                    return self
                if kind != Message.WEIGHTS:
                    raise ProtocolError("Expected expert weights or final assignment")
                item = identity(metadata.get("expert"))
                if item not in assigned:
                    raise ProtocolError("Received an unassigned expert")
                attempts[item] = attempts.get(item, 0) + 1
                if attempts[item] > 3:
                    raise ProtocolError("Exceeded expert transfer attempt limit")
                status, error = "loaded", None
                if item not in loaded:
                    try:
                        weights = decode_weights(payload)
                        if validate_weights(weights) != size:
                            raise ExpertRejected(
                                "Weights do not match assigned expert size"
                            )
                        expert = self.reconstruct(*item, metadata.get("model"), weights)
                        if (
                            not isinstance(expert, IdentifiedExpert)
                            or (expert.layer_id, expert.expert_id) != item
                        ):
                            raise ExpertRejected(
                                "Reconstructed expert has incorrect identity"
                            )
                        # Check the retained module, not just the received tensor bytes.
                        if validate_weights(expert.module.state_dict()) != size:
                            raise ExpertRejected(
                                "Reconstructed expert has incorrect storage size"
                            )
                        self.worker.register(expert)
                        loaded.add(item)
                    except ExpertRejected as exc:
                        status, error = "rejected", str(exc)
                    except (
                        ValueError,
                        KeyError,
                        TypeError,
                        RuntimeError,
                        MemoryError,
                        SafetensorError,
                    ) as exc:
                        status, error = "failed", f"{type(exc).__name__}: {exc}"
                    finally:
                        # Do not retain failed weights or an extra serialized expert.
                        weights = expert = None
                payload = b""
                send_message(
                    connection,
                    Message.RESULT,
                    {"expert": item, "status": status, "error": error},
                )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            SafetensorError,
        ) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.close()
            raise

    def _monitor(self):
        try:
            while not self._stop.is_set():
                metadata, _ = expect_message(self._control_connection, Message.PING)
                send_message(self._control_connection, Message.PONG, metadata)
        except (OSError, ValueError) as exc:
            self._end_session(exc)
        finally:
            self._end_session()
            self._control_connection.close()

    def _monitor_tensor(self):
        try:
            receive_message(self._tensor_connection)
            raise ProtocolError("Tensor execution messages are not implemented yet")
        except (OSError, ValueError) as exc:
            self._end_session(exc)
        finally:
            self._end_session()
            self._tensor_connection.close()

    def _end_session(self, error=None):
        with self._state_lock:
            if self._stop.is_set():
                return
            if error is not None:
                self.error = f"{type(error).__name__}: {error}"
            self.ready.clear()
            self.assignments = ()
            self.worker = ExpertWorker()
            for connection in (self._control_connection, self._tensor_connection):
                if connection is not None:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            self._stop.set()

    def wait_until_disconnected(self, timeout=None):
        """Wait for either channel to fail or for explicit shutdown."""
        if self._thread is None:
            raise RuntimeError("Worker has not initialized")
        return self._stop.wait(timeout)

    def close(self):
        self._end_session()
        for thread in (self._thread, self._tensor_thread):
            if thread is not None:
                thread.join()
        for connection in (self._control_connection, self._tensor_connection):
            if connection is not None:
                connection.close()

    def __enter__(self):
        return self.initialize()

    def __exit__(self, *args):
        self.close()

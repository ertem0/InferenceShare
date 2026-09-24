"""Memory-budgeted startup allocation and per-connection worker health."""

import math
import secrets
import socket
import threading
import uuid
from collections.abc import Callable, Iterable
from copy import deepcopy

from safetensors import SafetensorError
from torch import Tensor

from .protocol import (
    Message,
    ProtocolError,
    encode_weights,
    expect_message,
    identities,
    identity,
    receive_message,
    send_message,
    validate_weights,
)


class Coordinator:
    """Own startup assignments; prepare_expert supplies model-specific weights.

    Each control connection starts a fresh two-channel node session. Readiness means its final assignment
    loaded successfully and its tensor channel attached, not full-model coverage. Zero-expert
    assignments are allowed. Snapshots are copies protected by the state lock.
    """

    def __init__(
        self,
        expert_ids: Iterable[tuple[int, int]],
        expert_size_bytes: int,
        prepare_expert: Callable[[int, int], tuple[dict, dict[str, Tensor]]],
        *,
        address: tuple[str, int] = ("127.0.0.1", 0),
        tensor_address: tuple[str, int] | None = None,
        attachment_timeout: float = 10.0,
        initialization_timeout: float = 30.0,
        heartbeat_interval: float = 1.0,
        health_timeout: float = 5.0,
    ):
        ids = [identity(item) for item in expert_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate inventory identities")
        if type(expert_size_bytes) is not int or expert_size_bytes <= 0:
            raise ValueError("expert_size_bytes must be a positive integer")
        for value in (
            initialization_timeout,
            heartbeat_interval,
            health_timeout,
            attachment_timeout,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Timeouts and heartbeat interval must be positive and finite"
                )
        self.expert_size_bytes = expert_size_bytes
        self.prepare_expert = prepare_expert
        self.address = address
        self.tensor_address = tensor_address or (address[0], 0)
        self.attachment_timeout = attachment_timeout
        self.initialization_timeout = initialization_timeout
        self.heartbeat_interval = heartbeat_interval
        self.health_timeout = health_timeout
        self._inventory = {
            item: {"node_id": None, "status": "unallocated"} for item in sorted(ids)
        }
        self._nodes = {}
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._listeners = []
        self._accept_threads = []
        self._threads = []
        self._pending_connections = set()

    @property
    def inventory(self):
        with self._condition:
            return deepcopy(self._inventory)

    @property
    def nodes(self):
        """Return node snapshots with assignments derived only from the inventory."""
        with self._condition:
            nodes = {
                node_id: {
                    "memory_budget_bytes": node["memory_budget_bytes"],
                    "status": node["status"],
                    "error": node["error"],
                    "experts": [],
                }
                for node_id, node in self._nodes.items()
            }
            for expert_id, entry in self._inventory.items():
                if entry["node_id"] is not None:
                    nodes[entry["node_id"]]["experts"].append(expert_id)
            return nodes

    def wait_for_node(self, node_id, status, timeout=5.0):
        """Wait explicitly for ready/disconnected state; useful for callers and tests."""
        with self._condition:
            return self._condition.wait_for(
                lambda: self._nodes.get(node_id, {}).get("status") == status,
                timeout=timeout,
            )

    def start(self):
        if self._listeners or self._stop.is_set():
            raise RuntimeError("Coordinator has already started or closed")
        try:
            for address in (self.address, self.tensor_address):
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._listeners.append(listener)
                listener.bind(address)
                listener.listen()
                listener.settimeout(0.2)
        except OSError:
            for listener in self._listeners:
                listener.close()
            self._listeners.clear()
            raise
        self.address, self.tensor_address = [
            listener.getsockname() for listener in self._listeners
        ]
        for listener, handler in zip(
            self._listeners, (self._serve, self._serve_tensor), strict=True
        ):
            thread = threading.Thread(
                target=self._accept, args=(listener, handler), daemon=True
            )
            self._accept_threads.append(thread)
            thread.start()
        return self

    def _accept(self, listener, handler):
        while not self._stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            with self._condition:
                if self._stop.is_set():
                    connection.close()
                    return
                self._pending_connections.add(connection)
                thread = threading.Thread(
                    target=handler, args=(connection,), daemon=True
                )
                self._threads.append(thread)
                thread.start()

    def _reserve(self, node_id, budget, connection):
        with self._condition:
            if self._stop.is_set():
                raise ConnectionError("Coordinator is shutting down")
            available = [
                item
                for item, entry in self._inventory.items()
                if entry["node_id"] is None
            ]
            assigned = available[: budget // self.expert_size_bytes]
            for item in assigned:
                self._inventory[item] = {"node_id": node_id, "status": "loading"}
            self._nodes[node_id] = {
                "memory_budget_bytes": budget,
                "status": "loading",
                "error": None,
                "control_connection": connection,
                "tensor_connection": None,
                "session_token": secrets.token_hex(32),
            }
            self._pending_connections.discard(connection)
            self._condition.notify_all()
            return assigned

    def _serve(self, connection):
        node_id = uuid.uuid4().hex
        failure = None
        try:
            connection.settimeout(self.initialization_timeout)
            request, _ = expect_message(connection, Message.INITIALIZE)
            budget = request.get("memory_budget_bytes")
            if type(budget) is not int or budget < 0:
                raise ProtocolError("Memory budget must be a nonnegative integer")
            assigned = self._reserve(node_id, budget, connection)
            send_message(
                connection,
                Message.ASSIGN,
                {
                    "node_id": node_id,
                    "experts": assigned,
                    "expert_size_bytes": self.expert_size_bytes,
                    "heartbeat_interval": self.heartbeat_interval,
                    "health_timeout": self.health_timeout,
                },
            )
            successful = []
            for item in assigned:
                metadata, weights = self.prepare_expert(*item)
                if validate_weights(weights) != self.expert_size_bytes:
                    raise ValueError(
                        "Expert weights do not match the declared equal expert size"
                    )
                payload = encode_weights(weights)
                del weights
                for attempt in range(1, 4):
                    send_message(
                        connection,
                        Message.WEIGHTS,
                        {
                            "expert": item,
                            "model": metadata,
                            "attempt": attempt,
                        },
                        payload,
                    )
                    result, _ = expect_message(connection, Message.RESULT)
                    if identity(result.get("expert")) != item:
                        raise ProtocolError("Load report identifies the wrong expert")
                    status = result.get("status")
                    if status == "loaded":
                        successful.append(item)
                        break
                    if status not in ("failed", "rejected") or not isinstance(
                        result.get("error"), str
                    ):
                        raise ProtocolError("Invalid expert load result")
                    if status == "rejected":
                        break
                del payload
            # Release failures before asking the worker to confirm its final list.
            with self._condition:
                for item in set(assigned) - set(successful):
                    self._inventory[item] = {"node_id": None, "status": "unallocated"}
            send_message(connection, Message.FINALIZE, {"experts": successful})
            ready, _ = expect_message(connection, Message.READY)
            if identities(ready.get("experts")) != successful:
                raise ProtocolError("Worker readiness does not match final assignment")
            with self._condition:
                if (
                    self._stop.is_set()
                    or self._nodes[node_id]["status"] == "disconnected"
                ):
                    return
                self._nodes[node_id]["status"] = "initialized"
                token = self._nodes[node_id]["session_token"]
                self._condition.notify_all()
            send_message(
                connection,
                Message.CONFIRMED,
                {
                    "node_id": node_id,
                    "session_token": token,
                    "tensor_port": self.tensor_address[1],
                },
            )
            with self._condition:
                attached = self._condition.wait_for(
                    lambda: (
                        self._nodes[node_id]["status"] in ("ready", "disconnected")
                        or self._stop.is_set()
                    ),
                    timeout=self.attachment_timeout,
                )
                if not attached:
                    raise TimeoutError("Tensor connection attachment timed out")
                if self._nodes[node_id]["status"] != "ready" or self._stop.is_set():
                    return
            connection.settimeout(self.health_timeout)
            sequence = 0
            while not self._stop.wait(self.heartbeat_interval):
                send_message(connection, Message.PING, {"sequence": sequence})
                pong, _ = expect_message(connection, Message.PONG)
                if pong.get("sequence") != sequence:
                    raise ProtocolError("Invalid heartbeat acknowledgment")
                sequence += 1
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            RuntimeError,
            SafetensorError,
        ) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            try:
                send_message(connection, Message.ERROR, {"error": failure})
            except OSError:
                pass  # The peer may already have disconnected.
        finally:
            self._disconnect(node_id, failure)
            connection.close()
            with self._condition:
                self._pending_connections.discard(connection)

    def _serve_tensor(self, connection):
        node_id = None
        failure = None
        try:
            connection.settimeout(self.attachment_timeout)
            request, _ = expect_message(connection, Message.ATTACH_TENSOR)
            requested_id = request.get("node_id")
            token = request.get("session_token")
            if not isinstance(requested_id, str) or not isinstance(token, str):
                raise ProtocolError("Invalid tensor session identity")
            with self._condition:
                node = self._nodes.get(requested_id)
                if (
                    node is None
                    or node["status"] != "initialized"
                    or node["tensor_connection"] is not None
                    or not secrets.compare_digest(
                        token.encode(), node["session_token"].encode()
                    )
                ):
                    raise ProtocolError("Unknown, duplicate, or invalid tensor session")
                node_id = requested_id
                node["tensor_connection"] = connection
                self._pending_connections.discard(connection)
            send_message(connection, Message.TENSOR_ATTACHED, {"node_id": node_id})
            with self._condition:
                if node["status"] == "disconnected" or self._stop.is_set():
                    return
                node["status"] = "ready"
                for entry in self._inventory.values():
                    if entry["node_id"] == node_id:
                        entry["status"] = "ready"
                self._condition.notify_all()
            # No inference messages exist yet. Own this channel's reader so EOF
            # tears down the whole session; later milestones add dispatch here.
            connection.settimeout(None)
            receive_message(connection)
            raise ProtocolError("Tensor execution messages are not implemented yet")
        except (OSError, ValueError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            try:
                send_message(connection, Message.ERROR, {"error": failure})
            except OSError:
                pass
        finally:
            if node_id is not None:
                self._disconnect(node_id, failure)
            connection.close()
            with self._condition:
                self._pending_connections.discard(connection)

    def _disconnect(self, node_id, failure):
        """Release one session once; wake both readers without closing their sockets."""
        with self._condition:
            node = self._nodes.get(node_id)
            if node is None or node["status"] == "disconnected":
                return
            for entry in self._inventory.values():
                if entry["node_id"] == node_id:
                    entry.update(node_id=None, status="unallocated")
            node.update(status="disconnected", error=failure, session_token=None)
            for channel in ("control_connection", "tensor_connection"):
                connection = node[channel]
                if connection is not None:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass  # Its reader may already have observed disconnection.
                    node[channel] = None
            self._condition.notify_all()

    def close(self):
        self._stop.set()
        for listener in self._listeners:
            listener.close()
        with self._condition:
            for connection in self._pending_connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            for node_id in self._nodes:
                self._disconnect(node_id, None)
            self._condition.notify_all()
        for thread in self._accept_threads:
            thread.join()
        for thread in self._threads:
            thread.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()

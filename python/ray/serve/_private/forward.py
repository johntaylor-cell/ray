"""Forwarding: a deployment that returns a child's response unawaited hands it to its caller.

With splice, the parent's Serve layer sends the unsent child call to the child replica's
HTTP port and copies the child's response bytes into the caller's connection, so user
code and per-chunk Python objects stay out of the response path.
"""

import asyncio
import functools
import logging
from typing import Any, Dict, List, Optional, Tuple, cast

import starlette.responses

import ray
from ray import cloudpickle
from ray._private.authentication import authentication_utils
from ray._private.authentication.authentication_constants import (
    AUTHORIZATION_HEADER_NAME,
    RAY_AUTHORIZATION_HEADER_NAME,
)
from ray._private.authentication.http_token_authentication import (
    get_auth_headers_if_auth_enabled,
)
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.http_util import convert_object_to_asgi_messages

logger = logging.getLogger(SERVE_LOGGER_NAME)

# Served only on replicas HAProxy never routes to, since the body is a pickled call.
FORWARD_ROUTE = "/-/serve/forward"
# Set by a child that turned a call away, so the parent tries elsewhere.
REFUSED_HEADER = b"x-serve-forward-refused"
REFUSED_CAPACITY, REFUSED_INGRESS = b"capacity", b"ingress"
# The child's last chunk; held back so uvicorn ends the caller's response itself.
TERMINATOR = b"0\r\n\r\n"
HOP_BY_HOP = {
    b"connection",
    b"keep-alive",
    b"transfer-encoding",
    b"content-length",
    b"te",
    b"trailer",
    b"upgrade",
}
# Past HIGH_WATER bytes buffered for a slow caller, the parent stops reading from the child.
HIGH_WATER, LOW_WATER = 256 * 1024, 64 * 1024
# When reading resumes the caller still has LOW_WATER bytes queued, far more than a poll's worth.
RESUME_POLL_S = 0.01
# Child replicas to try before sending the call through the handle instead.
FORWARD_ATTEMPTS = 3
# SSE field names; a stream whose first chunk starts with one is served as SSE.
SSE_PREFIXES = (b"data:", b"event:", b"id:", b"retry:", b":")
# How each forward was delivered; read by tests.
COUNTS = {"splice": 0, "relay": 0}


def is_forwardable(result: Any) -> bool:
    from ray.serve.handle import DeploymentResponse, DeploymentResponseGenerator

    return isinstance(result, (DeploymentResponse, DeploymentResponseGenerator))


def chunk_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, bytes):
        return chunk
    if isinstance(chunk, str):
        return chunk.encode("utf-8")
    raise TypeError(
        "A forwarded stream must yield str or bytes, got " f"{type(chunk).__name__}."
    )


def stream_content_type(first: Any) -> bytes:
    """The content type a stream's first chunk implies."""
    if first is None:
        return b"text/plain; charset=utf-8"
    if chunk_bytes(first).startswith(SSE_PREFIXES):
        return b"text/event-stream"
    if isinstance(first, str):
        return b"text/plain; charset=utf-8"
    return b"application/octet-stream"


def _held_back(data: bytes) -> int:
    """How many trailing bytes could be the start of the final chunk and must wait."""
    for n in range(len(TERMINATOR), 0, -1):
        if data.endswith(TERMINATOR[:n]):
            return n
    return 0


def _find_transport(send):
    """uvicorn's client socket behind Serve's send wrappers, or None.

    None means the caller is not a direct HTTP connection, for example the Serve proxy.
    """
    stack, seen = [send], set()
    while stack:
        fn = stack.pop()
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        owner = getattr(fn, "__self__", None)
        # Only uvicorn's per-request cycle holds the client socket; nothing else qualifies.
        if (
            owner is not None
            and type(owner).__name__ == "RequestResponseCycle"
            and type(owner).__module__.startswith("uvicorn.")
        ):
            return owner.transport
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                content = cell.cell_contents
            except ValueError:
                continue
            if callable(content):
                stack.append(content)
    return None


class _ChunkDecoder:
    """Unframes chunked bytes when the caller has no socket and ASGI frames them again."""

    def __init__(self):
        self.buf = bytearray()
        self.need: Optional[int] = None

    def feed(self, data: bytes) -> List[bytes]:
        self.buf += data
        payloads = []
        while True:
            if self.need is None:
                end = self.buf.find(b"\r\n")
                if end < 0:
                    break
                size = int(bytes(self.buf[:end]).split(b";")[0], 16)
                del self.buf[: end + 2]
                self.need = size + 2
            if len(self.buf) < self.need:
                break
            if self.need > 2:
                payloads.append(bytes(self.buf[: self.need - 2]))
            del self.buf[: self.need]
            self.need = None
        return payloads


class _QueueOut:
    """Stands in for the caller's socket when there is none; drained through ASGI."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.size = 0

    def write(self, data: bytes):
        self.size += len(data)
        self.queue.put_nowait(data)

    def is_closing(self) -> bool:
        return False

    def get_write_buffer_size(self) -> int:
        return self.size


class _Upstream(asyncio.Protocol):
    """One kept-alive child connection; each response's body passes through unparsed."""

    transport: asyncio.Transport

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self.begin()

    def begin(self):
        # A pooled connection carries many responses, so per-response state starts here.
        self.buf = bytearray()
        self.head_ready = self.loop.create_future()
        self.done = self.loop.create_future()
        self.status = 502
        self.headers: List[Tuple[bytes, bytes]] = []
        self.refused: Optional[bytes] = None
        self.chunked = False
        self.length: Optional[int] = None
        self.out = None
        self.tail = b""
        self.paused = False
        self.complete = False

    def connection_made(self, transport):
        self.transport = cast(asyncio.Transport, transport)

    def body(self) -> bytes:
        return bytes(self.buf if self.length is None else self.buf[: self.length])

    def attach(self, out):
        """Flush what arrived before the caller's socket was known, then write reads straight to it."""
        pending = bytes(self.buf)
        self.buf = bytearray()
        self._forward(out, pending)
        if self.tail == TERMINATOR:
            self.finish()
        else:
            # No await since the flush above, so no read can slip in between.
            self.out = out

    def _forward(self, out, data: bytes):
        # Hold back only what could begin the final chunk, so each complete chunk goes out at once.
        if self.tail:
            data = self.tail + data
        held = _held_back(data)
        if held:
            self.tail = data[-held:]
            data = data[:-held]
        else:
            self.tail = b""
        if data:
            out.write(data)

    def finish(self):
        self.complete = True
        if not self.done.done():
            self.done.set_result(None)

    def data_received(self, data: bytes):
        out = self.out
        if out is not None:
            # Hot path: one callback per socket read, however many chunks it holds.
            if out.is_closing():
                # Closing the upstream is what tells the child to stop.
                self.transport.close()
                return
            self._forward(out, data)
            if self.tail == TERMINATOR:
                self.finish()
            elif out.get_write_buffer_size() > HIGH_WATER:
                self.pause()
            return
        self.buf += data
        while not self.head_ready.done():
            end = self.buf.find(b"\r\n\r\n")
            if end < 0:
                return
            head = bytes(self.buf[:end]).split(b"\r\n")
            del self.buf[: end + 4]
            status = int(head[0].split(b" ", 2)[1])
            if 100 <= status < 200:
                # Interim responses such as 100 Continue come before the real one.
                continue
            self.status = status
            for line in head[1:]:
                name, _, value = line.partition(b":")
                name, value = name.strip().lower(), value.strip()
                if name == b"transfer-encoding" and value.lower() == b"chunked":
                    self.chunked = True
                elif name == b"content-length":
                    self.length = int(value)
                elif name == REFUSED_HEADER:
                    self.refused = value
                elif name not in HOP_BY_HOP:
                    self.headers.append((name, value))
            self.head_ready.set_result(None)
        if (
            not self.chunked
            and self.length is not None
            and len(self.buf) >= self.length
        ):
            self.finish()

    def pause(self):
        if not self.paused:
            self.paused = True
            self.transport.pause_reading()
            self.loop.call_later(RESUME_POLL_S, self.resume_when_drained)

    def resume_when_drained(self):
        if self.out is None or self.closed:
            return
        if self.out.is_closing():
            self.transport.close()
        elif self.out.get_write_buffer_size() > LOW_WATER:
            self.loop.call_later(RESUME_POLL_S, self.resume_when_drained)
        else:
            self.paused = False
            self.transport.resume_reading()

    def connection_lost(self, exc):
        self.closed = True
        if not self.head_ready.done():
            self.head_ready.set_exception(
                exc or ConnectionError("child closed before the response head")
            )
        if not self.done.done():
            self.done.set_result(None)


# Idle child connections by (host, port); a replica runs one event loop.
_idle: Dict[Tuple[str, int], List[_Upstream]] = {}


@functools.lru_cache(maxsize=None)
def _local_node_id() -> str:
    return ray.get_runtime_context().get_node_id()


def _dial_address(selection) -> Optional[Tuple[str, int]]:
    endpoint = selection._replica.backend_http_endpoint
    if endpoint is None:
        return None
    host, port = endpoint
    # Replica servers bind to loopback unless HAProxy needs every interface.
    if selection.node_id == _local_node_id():
        host = "127.0.0.1"
    return host, port


async def _connect(endpoint: Tuple[str, int]) -> Tuple[_Upstream, bool]:
    idle = _idle.get(endpoint)
    while idle:
        up = idle.pop()
        if not up.closed and not up.transport.is_closing():
            return up, True
    _, up = await asyncio.get_running_loop().create_connection(_Upstream, *endpoint)
    return up, False


def _release(endpoint: Tuple[str, int], up: _Upstream):
    if up.complete and not up.closed:
        if up.paused:
            up.transport.resume_reading()
        up.begin()
        _idle.setdefault(endpoint, []).append(up)
    else:
        up.transport.close()


def _request_bytes(body: bytes) -> bytes:
    lines = [
        b"POST " + FORWARD_ROUTE.encode() + b" HTTP/1.1",
        b"host: serve",
        b"content-type: application/octet-stream",
        b"content-length: %d" % len(body),
    ]
    for name, value in get_auth_headers_if_auth_enabled({}).items():
        lines.append(name.encode() + b": " + value.encode())
    return b"\r\n".join(lines) + b"\r\n\r\n" + body


async def _open(endpoint: Tuple[str, int], request: bytes) -> Optional[_Upstream]:
    """Send the call and wait for the head; None if the replica could not take it."""
    try:
        up, reused = await _connect(endpoint)
    except OSError:
        return None
    up.transport.write(request)
    try:
        await up.head_ready
    except ConnectionError:
        # A pooled connection the child closed while idle, or a child that just died.
        up.transport.close()
        if reused:
            return await _open(endpoint, request)
        return None
    return up


async def _deliver(up: _Upstream, send) -> None:
    if not up.chunked:
        # A whole body arrives as one piece, so it goes through ASGI.
        await up.done
        await send(
            {"type": "http.response.start", "status": up.status, "headers": up.headers}
        )
        await send({"type": "http.response.body", "body": up.body()})
        return
    await send(
        {
            "type": "http.response.start",
            "status": up.status,
            "headers": up.headers + [(b"transfer-encoding", b"chunked")],
        }
    )
    out = _find_transport(send)
    if out is not None:
        up.attach(out)
        await up.done
    else:
        await _relay_chunks(up, send)
    if up.tail != TERMINATOR:
        # Raising aborts the caller's connection instead of ending a cut-off stream cleanly.
        raise ConnectionError("The child's response ended before its final chunk.")
    await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _relay_chunks(up: _Upstream, send) -> None:
    out, decoder = _QueueOut(), _ChunkDecoder()
    up.done.add_done_callback(lambda _: out.queue.put_nowait(None))
    up.attach(out)
    while (data := await out.queue.get()) is not None:
        out.size -= len(data)
        for payload in decoder.feed(data):
            await send(
                {"type": "http.response.body", "body": payload, "more_body": True}
            )


class PendingForward:
    """A child response a handler returned unawaited, delivered as the handler's response.

    The call is claimed at once, before the handler's loop can turn and route it.
    """

    def __init__(self, response):
        self.response = response
        self.call = response._claim_pending_call()

    async def deliver(self, scope, receive, send) -> None:
        if self.call is None:
            # Routing already started, so the result comes back over the handle.
            await relay(self.response, scope, receive, send)
            return
        handle, args, kwargs = self.call
        await _splice(handle, args, kwargs, scope, receive, send)


async def _splice(handle, args, kwargs, scope, receive, send) -> None:
    for _ in range(FORWARD_ATTEMPTS):
        async with handle._choose_replica(
            args, dict(kwargs, _reserve=False)
        ) as selection:
            endpoint = _dial_address(selection)
            if endpoint is None:
                break
            try:
                body = cloudpickle.dumps((selection._request_metadata, args, kwargs))
            except Exception:
                # Arguments such as an unresolved DeploymentResponse only travel by handle.
                break
            up = await _open(endpoint, _request_bytes(body))
            if up is None:
                continue
            if up.refused is not None:
                # The refusal's body is chunked, so it is never read to the end.
                up.transport.close()
                if up.refused == REFUSED_CAPACITY:
                    continue
                break
            COUNTS["splice"] += 1
            try:
                await _deliver(up, send)
            finally:
                _release(endpoint, up)
            return
    await relay(handle.remote(*args, **kwargs), scope, receive, send)


async def relay(response, scope, receive, send) -> None:
    """Deliver a response that travels over the handle; Serve code, not the user's."""
    from ray.serve.handle import DeploymentResponseGenerator

    COUNTS["relay"] += 1
    if not isinstance(response, DeploymentResponseGenerator):
        result = await response
        if isinstance(result, starlette.responses.Response):
            await result(scope, receive, send)
        else:
            for message in convert_object_to_asgi_messages(result):
                await send(message)
        return
    results = response.__aiter__()
    try:
        first = await results.__anext__()
    except StopAsyncIteration:
        first = None
    await _send_stream(first, results, send, None)


async def _send_stream(first, results, send, status_code_callback) -> None:
    if status_code_callback is not None:
        status_code_callback("200")
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", stream_content_type(first))],
        }
    )
    if first is not None:
        await send(
            {
                "type": "http.response.body",
                "body": chunk_bytes(first),
                "more_body": True,
            }
        )
        async for chunk in results:
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk_bytes(chunk),
                    "more_body": True,
                }
            )
    await send({"type": "http.response.body", "body": b"", "more_body": False})


def _authorized(scope) -> bool:
    if not authentication_utils.is_token_auth_enabled():
        return True
    headers = dict(scope["headers"])
    token = headers.get(AUTHORIZATION_HEADER_NAME.encode()) or headers.get(
        RAY_AUTHORIZATION_HEADER_NAME.encode()
    )
    return token is not None and authentication_utils.validate_request_token(
        token.decode()
    )


async def _read_body(receive) -> bytes:
    body = bytearray()
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            return bytes(body)


async def _wait_for_disconnect(receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


async def _send_simple(send, status: int, text: str, refused: Optional[bytes] = None):
    headers = [(REFUSED_HEADER, refused)] if refused is not None else []
    for message in convert_object_to_asgi_messages(
        text, status_code=status, extra_headers=headers
    ):
        await send(message)


async def _answer(
    wrapper, request_metadata, args, kwargs, scope, receive, send, callback
):
    try:
        if request_metadata.is_streaming:
            results = wrapper.call_user_generator(request_metadata, args, kwargs)
            try:
                first = await results.__anext__()
            except StopAsyncIteration:
                first = None
            await _send_stream(first, results, send, callback)
            return
        result = await wrapper.call_user_method(request_metadata, args, kwargs)
    except Exception as e:
        response = wrapper.handle_exception(e)
        callback(str(response.status_code))
        await response(scope, receive, send)
        raise
    if is_forwardable(result):
        await PendingForward(result).deliver(scope, receive, send)
    elif isinstance(result, starlette.responses.Response):
        await result(scope, receive, send)
    else:
        for message in convert_object_to_asgi_messages(result):
            await send(message)


async def serve_forwarded_call(replica, scope, receive, send) -> None:
    """Run a call a parent replica forwarded here and answer it as HTTP."""
    if replica._ingress or replica._is_ingress_request_router:
        # HAProxy routes client traffic to these replicas, so they never take pickled calls.
        await _send_simple(send, 404, "Not found", REFUSED_INGRESS)
        return
    if not _authorized(scope):
        await _send_simple(send, 401, "Unauthorized")
        return
    request_metadata, args, kwargs = cloudpickle.loads(await _read_body(receive))
    limit = replica.max_queued_requests
    if replica._quiescing or (limit != -1 and replica._num_queued_requests >= limit):
        # Queue like a direct-ingress request, but let the parent try another replica.
        await _send_simple(send, 503, "Replica unavailable", REFUSED_CAPACITY)
        return
    finished = False

    async def tracked_send(message):
        nonlocal finished
        await send(message)
        if message["type"] == "http.response.body" and not message.get("more_body"):
            finished = True

    wrapper = replica._user_callable_wrapper
    try:
        with replica._wrap_request(
            request_metadata
        ) as callback, replica._track_queued_request() as release_queue_slot:
            async with replica._start_request(request_metadata):
                release_queue_slot()
                answer = asyncio.ensure_future(
                    _answer(
                        wrapper,
                        request_metadata,
                        args,
                        kwargs,
                        scope,
                        receive,
                        tracked_send,
                        callback,
                    )
                )
                disconnect = asyncio.ensure_future(_wait_for_disconnect(receive))
                await asyncio.wait(
                    (answer, disconnect), return_when=asyncio.FIRST_COMPLETED
                )
                disconnect.cancel()
                if not answer.done() and not finished:
                    # The parent hung up, which is how a caller's disconnect reaches the child.
                    callback("499")
                    answer.cancel()
                await answer
    except Exception:
        # A sent error response is already recorded; a cut-off stream must abort.
        if not finished:
            raise

import asyncio
import sys

import pytest

import ray.serve.context
from ray.serve._private.common import RequestMetadata
from ray.serve._private.forward import (
    FORWARD_ROUTE,
    HIGH_WATER,
    REFUSED_CAPACITY,
    REFUSED_HEADER,
    TERMINATOR,
    _answer,
    _ChunkDecoder,
    _find_transport,
    _held_back,
    _idle,
    _relay_chunks,
    _release,
    _request_bytes,
    _Upstream,
    chunk_bytes,
    stream_content_type,
)

HEAD = (
    b"HTTP/1.1 200 OK\r\n"
    b"content-type: text/event-stream\r\n"
    b"transfer-encoding: chunked\r\n"
    b"connection: keep-alive\r\n\r\n"
)


def _chunk(payload: bytes) -> bytes:
    return b"%x\r\n%s\r\n" % (len(payload), payload)


def _events(n: int):
    return [_chunk(b'data: {"i": %d}\n\n' % i) for i in range(n)]


class _Transport:
    def __init__(self):
        self.written = bytearray()
        self.closing = False
        self.reading = True
        self.buffered = 0

    def write(self, data):
        self.written += data

    def is_closing(self):
        return self.closing

    def close(self):
        self.closing = True

    def pause_reading(self):
        self.reading = False

    def resume_reading(self):
        self.reading = True

    def get_write_buffer_size(self):
        return self.buffered


def _upstream() -> _Upstream:
    up = _Upstream()
    up.connection_made(_Transport())
    return up


@pytest.mark.parametrize(
    "first,content_type",
    [
        ("data: hi\n\n", b"text/event-stream"),
        (b"event: done\n\n", b"text/event-stream"),
        (": keep-alive\n\n", b"text/event-stream"),
        ("hello", b"text/plain; charset=utf-8"),
        (b"\x00\x01", b"application/octet-stream"),
        (None, b"text/plain; charset=utf-8"),
    ],
)
def test_stream_content_type_comes_from_the_first_chunk(first, content_type):
    assert stream_content_type(first) == content_type


def test_chunk_bytes_accepts_only_text_and_bytes():
    assert chunk_bytes("é") == "é".encode()
    assert chunk_bytes(b"x") == b"x"
    with pytest.raises(TypeError):
        chunk_bytes({"a": 1})


def test_request_bytes_post_the_call_to_the_forward_route():
    head, _, body = _request_bytes(b"pickled").partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    assert lines[0] == b"POST " + FORWARD_ROUTE.encode() + b" HTTP/1.1"
    assert b"content-length: 7" in lines
    assert body == b"pickled"


def test_streamed_body_passes_through_without_terminator():
    async def run():
        up = _upstream()
        events = _events(3)
        up.data_received(HEAD + events[0][:7])
        assert up.head_ready.done() and up.chunked and up.status == 200
        names = [name for name, _ in up.headers]
        assert b"transfer-encoding" not in names and b"connection" not in names
        caller = _Transport()
        up.attach(caller)
        rest = events[0][7:] + b"".join(events[1:]) + TERMINATOR
        # 3-byte reads split events and the terminator across callbacks.
        for i in range(0, len(rest), 3):
            up.data_received(rest[i : i + 3])
        assert up.complete and up.done.done()
        assert bytes(caller.written) == b"".join(events)

    asyncio.run(run())


def test_complete_chunks_are_written_as_soon_as_they_are_read():
    async def run():
        up = _upstream()
        up.data_received(HEAD)
        caller = _Transport()
        up.attach(caller)
        for event in _events(3):
            up.data_received(event)
            assert bytes(caller.written).endswith(event)
        assert up.tail == b""

    asyncio.run(run())


def test_held_back_only_holds_a_possible_final_chunk():
    assert _held_back(_events(1)[0]) == 0
    assert _held_back(b"...\r\n0") == 1
    assert _held_back(b"...\r\n0\r\n\r") == 4
    assert _held_back(b"...\r\n" + TERMINATOR) == 5
    # A chunk size ending in 0 looks like the start of the terminator until more bytes arrive.
    assert _held_back(b"...\r\na0\r\n") == 3


def test_interim_1xx_responses_are_skipped():
    async def run():
        up = _upstream()
        event = _events(1)[0]
        up.data_received(b"HTTP/1.1 100 Continue\r\n\r\n" + HEAD + event)
        assert up.head_ready.done() and up.status == 200 and up.chunked
        assert bytes(up.buf) == event

    asyncio.run(run())


def test_refusal_is_read_from_the_head():
    async def run():
        up = _upstream()
        up.data_received(
            b"HTTP/1.1 503 Service Unavailable\r\n"
            + REFUSED_HEADER
            + b": "
            + REFUSED_CAPACITY
            + b"\r\ncontent-length: 2\r\n\r\nno"
        )
        assert up.status == 503 and up.refused == REFUSED_CAPACITY and up.complete
        assert REFUSED_HEADER not in [name for name, _ in up.headers]

    asyncio.run(run())


def test_whole_response_completes_on_content_length():
    async def run():
        up = _upstream()
        body = b'{"echo": "hi"}'
        head = b"HTTP/1.1 200 OK\r\ncontent-length: %d\r\n\r\n" % len(body)
        up.data_received(head + body[:5])
        assert up.head_ready.done() and not up.chunked and not up.complete
        up.data_received(body[5:])
        assert up.complete and up.body() == body

    asyncio.run(run())


def test_backpressure_pauses_reading_until_the_caller_drains():
    async def run():
        up = _upstream()
        up.data_received(HEAD)
        caller = _Transport()
        up.attach(caller)
        caller.buffered = HIGH_WATER + 1
        up.data_received(_events(1)[0])
        assert up.paused and not up.transport.reading
        caller.buffered = 0
        await asyncio.sleep(0.05)
        assert not up.paused and up.transport.reading

    asyncio.run(run())


def test_caller_disconnect_closes_the_upstream():
    async def run():
        up = _upstream()
        up.data_received(HEAD)
        caller = _Transport()
        up.attach(caller)
        caller.closing = True
        up.data_received(_events(1)[0])
        assert up.transport.closing and not caller.written

    asyncio.run(run())


def test_connection_lost_before_the_head_fails_head_ready():
    async def run():
        up = _upstream()
        up.connection_lost(None)
        with pytest.raises(ConnectionError):
            await up.head_ready
        assert up.closed and up.done.done()

    asyncio.run(run())


class RequestResponseCycle:
    # Stands in for uvicorn's per-request cycle, the only owner _find_transport accepts.
    __module__ = "uvicorn.protocols.http.httptools_impl"

    def __init__(self):
        self.transport = _Transport()

    async def send(self, message):
        pass


def _wrap(inner):
    async def send(message):
        await inner(message)

    return send


def test_find_transport_through_send_wrappers():
    cycle = RequestResponseCycle()
    assert _find_transport(_wrap(_wrap(cycle.send))) is cycle.transport
    assert _find_transport(_wrap(lambda message: None)) is None


def test_find_transport_ignores_sockets_uvicorn_does_not_own():
    async def run():
        # The parent's own upstream socket must never be mistaken for the caller's.
        up = _upstream()
        assert _find_transport(_wrap(up.pause)) is None

    asyncio.run(run())


def test_chunk_decoder_unframes_split_chunks():
    decoder = _ChunkDecoder()
    stream = b"".join(_events(3))
    payloads = []
    for i in range(0, len(stream), 4):
        payloads += decoder.feed(stream[i : i + 4])
    assert payloads == [b'data: {"i": %d}\n\n' % i for i in range(3)]


def test_caller_without_a_socket_gets_asgi_messages():
    async def run():
        up = _upstream()
        up.data_received(HEAD)
        messages = []

        async def send(message):
            messages.append(message)

        loop = asyncio.get_running_loop()
        for data in _events(3) + [TERMINATOR]:
            loop.call_soon(up.data_received, data)
        await asyncio.wait_for(_relay_chunks(up, send), timeout=5)
        assert up.complete
        assert b"".join(m["body"] for m in messages) == b"".join(
            b'data: {"i": %d}\n\n' % i for i in range(3)
        )
        assert all(m["more_body"] for m in messages)

    asyncio.run(run())


def test_release_pools_only_completed_connections():
    async def run():
        endpoint = ("10.0.0.1", 1)
        done, cut = _upstream(), _upstream()
        done.data_received(HEAD + TERMINATOR)
        done.attach(_Transport())
        _release(endpoint, done)
        _release(endpoint, cut)
        assert _idle.pop(endpoint) == [done] and not done.transport.closing
        assert cut.transport.closing

    asyncio.run(run())


def test_a_forwarded_handler_can_forward_again():
    seen = []

    class Wrapper:
        async def call_user_method(self, request_metadata, args, kwargs):
            seen.append(ray.serve.context._get_serve_request_context()._forwardable)
            return "ok"

    async def run():
        messages = []

        async def send(message):
            messages.append(message)

        meta = RequestMetadata(request_id="r", internal_request_id="i")
        await _answer(Wrapper(), meta, (), {}, {}, None, send, lambda code: None)
        return messages

    messages = asyncio.run(run())
    assert seen == [True]
    assert messages[-1]["type"] == "http.response.body"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))

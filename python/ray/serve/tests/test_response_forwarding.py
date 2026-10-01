import asyncio
import sys

import httpx
import pytest

import ray
from ray import serve
from ray._common.test_utils import SignalActor, wait_for_condition
from ray.serve._private.constants import (
    RAY_SERVE_ENABLE_DIRECT_INGRESS,
    SERVE_DEFAULT_APP_NAME,
)
from ray.serve._private.forward import FORWARD_ROUTE, REFUSED_HEADER, REFUSED_INGRESS
from ray.serve._private.test_utils import get_application_url
from ray.serve.handle import DeploymentHandle

pytestmark = pytest.mark.skipif(
    not RAY_SERVE_ENABLE_DIRECT_INGRESS,
    reason="Forwarding reads the child's HTTP port, which needs direct ingress.",
)


@pytest.fixture(scope="module")
def serve_cluster():
    ray.init(address="local", num_cpus=8, include_dashboard=False)
    serve.start()
    yield
    serve.shutdown()
    ray.shutdown()


@serve.deployment
class Gen:
    def __init__(self, signal=None):
        self.signal = signal

    async def stream(self, prompt: str):
        for word in prompt.split():
            yield f"data: {word}\n\n"

    async def words(self, prompt: str):
        for word in prompt.split():
            yield word + " "

    async def unary(self, prompt: str):
        return {"echo": prompt}

    async def fail(self, prompt: str):
        raise ValueError("boom")

    async def slow(self, prompt: str):
        try:
            for i in range(1000):
                yield f"data: {i}\n\n"
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            await self.signal.send.remote()
            raise


@serve.deployment
class Gateway:
    def __init__(self, gen: DeploymentHandle):
        self.gen = gen

    async def __call__(self, request):
        mode = request.query_params.get("mode", "stream")
        prompt = request.query_params.get("prompt", "")
        if mode == "stats":
            from ray.serve._private import forward

            return dict(forward.COUNTS)
        if mode == "late":
            response = self.gen.options(stream=True).stream.remote(prompt)
            # Routing starts while the handler waits, so Serve must relay instead.
            await asyncio.sleep(0.01)
            return response
        if mode in ("stream", "words", "slow"):
            return getattr(self.gen.options(stream=True), mode).remote(prompt)
        return getattr(self.gen, mode).remote(prompt)


def _url(app_name: str = SERVE_DEFAULT_APP_NAME) -> str:
    # The replica's own port, which is what HAProxy sends requests to.
    return get_application_url(app_name=app_name, from_proxy_manager=True)


def _get(url: str, **params) -> httpx.Response:
    return httpx.get(url + "/", params=params, timeout=30)


def _counts(url: str) -> dict:
    return _get(url, mode="stats").json()


@pytest.fixture(scope="module")
def gateway(serve_cluster):
    signal = SignalActor.remote()
    serve.run(Gateway.bind(Gen.bind(signal)))
    wait_for_condition(lambda: _get(_url(), mode="stats").status_code == 200)
    yield _url(), signal
    serve.delete(SERVE_DEFAULT_APP_NAME)


def test_stream_is_spliced_and_served_as_sse(gateway):
    url, _ = gateway
    before = _counts(url)
    response = _get(url, prompt="a b c")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert response.text == "data: a\n\ndata: b\n\ndata: c\n\n"
    assert _counts(url)["splice"] == before["splice"] + 1


def test_plain_stream_is_served_as_text(gateway):
    url, _ = gateway
    response = _get(url, mode="words", prompt="a b")
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert response.text == "a b "


def test_unary_result_is_spliced_as_json(gateway):
    url, _ = gateway
    before = _counts(url)
    response = _get(url, mode="unary", prompt="hi")
    assert response.status_code == 200
    assert response.json() == {"echo": "hi"}
    assert _counts(url)["splice"] == before["splice"] + 1


def test_call_routed_before_return_is_relayed(gateway):
    url, _ = gateway
    before = _counts(url)
    response = _get(url, mode="late", prompt="x y")
    assert response.text == "data: x\n\ndata: y\n\n"
    after = _counts(url)
    assert after["relay"] == before["relay"] + 1
    assert after["splice"] == before["splice"]


def test_child_error_is_a_500(gateway):
    url, _ = gateway
    assert _get(url, mode="fail").status_code == 500


def test_caller_disconnect_cancels_the_child(gateway):
    url, signal = gateway
    with httpx.stream("GET", url + "/", params={"mode": "slow"}, timeout=30) as stream:
        next(stream.iter_raw())
    ray.get(signal.wait.remote(), timeout=30)


def test_forward_through_the_serve_proxy(gateway):
    url, _ = gateway
    before = _counts(url)
    # The proxy calls the gateway through a handle, so there is no socket to splice into.
    response = httpx.get(
        get_application_url() + "/", params={"prompt": "p q"}, timeout=30
    )
    assert response.text == "data: p\n\ndata: q\n\n"
    assert _counts(url)["splice"] == before["splice"] + 1


def test_ingress_replicas_refuse_forwarded_calls(gateway):
    url, _ = gateway
    # HAProxy routes client traffic here, so a pickled call must never run.
    response = httpx.post(url + FORWARD_ROUTE, content=b"not a call", timeout=30)
    assert response.status_code == 404
    assert response.headers[REFUSED_HEADER.decode()] == REFUSED_INGRESS.decode()


def test_forward_frees_the_parent_slot_while_streaming(serve_cluster):
    # One slot: if the open stream held it, the second request would wait for the stream.
    serve.run(
        Gateway.options(max_ongoing_requests=1).bind(Gen.bind(SignalActor.remote())),
        name="slot",
        route_prefix="/slot",
    )
    try:
        url = _url("slot")
        wait_for_condition(lambda: _get(url, mode="stats").status_code == 200)
        with httpx.stream("GET", url + "/", params={"mode": "slow"}, timeout=30) as s:
            next(s.iter_raw())
            assert _get(url, mode="stats").status_code == 200
    finally:
        serve.delete("slot")


def test_forward_to_another_apps_ingress_is_relayed(serve_cluster):
    serve.run(Gen.bind(), name="child", route_prefix="/child")
    serve.run(
        Gateway.bind(serve.get_deployment_handle("Gen", app_name="child")),
        name="gw",
        route_prefix="/gw",
    )
    try:
        url = _url("gw")
        wait_for_condition(lambda: _get(url, mode="stats").status_code == 200)
        before = _counts(url)
        assert _get(url, prompt="m n").text == "data: m\n\ndata: n\n\n"
        assert _counts(url)["relay"] == before["relay"] + 1
    finally:
        serve.delete("gw")
        serve.delete("child")


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))

"""The control page's web server over real HTTP: a `ControlServer` on 127.0.0.1, port 0, sent real
requests, with a stub controller attached that records the commands it's given, so a refused
request can be seen never to reach it.

Also `resolve_bind`, which decides where the page listens and whether it goes behind HTTPS, and
`route_warning`, which the route watch asks while it runs, each on `tailscale serve status --json`
output as Tailscale prints it, with the Tailscale CLI faked.
"""

import contextlib
import http.client
import io
import json
import socket
import time
import unittest
from concurrent.futures import Future
from unittest import mock

from fakes import DeadStream, control

from so_101.control import server
from so_101.control.commands import Result
from so_101.control.server import Bind, ControlServer

TS_NAME = "studio.tail1234.ts.net"
LOGIN = "alice@example.com"


class StubController:
    """Accepts every command at once and records it. Its status is fixed."""

    def __init__(self) -> None:
        self.commands: list[dict] = []
        self.heartbeats: list[tuple] = []
        self.previews = None

    def submit(self, command: str, source: str, *, client: str | None = None,
               who: str | None = None, **options) -> Future:
        self.commands.append({"command": command, "source": source, "client": client, "who": who, **options})
        future: Future = Future()
        future.set_result(Result(command, True))
        return future

    def heartbeat(self, client: str, visible: bool = True) -> None:
        self.heartbeats.append((client, visible))

    def status(self, client: str | None = None) -> dict:
        return {"state": "resting", "last_event": 0, "controller": None}

    def events(self, after: int, limit: int) -> list[dict]:
        return []


class Server:
    """A started ControlServer with `controller` attached, and requests to it."""

    def __init__(self, test: unittest.TestCase, controller, https: str | None = None,
                 warnings: tuple[str, ...] = (), routes=None) -> None:
        """`routes` is what the server asks every ROUTES_EVERY_S on its own thread (`route_warning`)."""
        self.server = ControlServer(Bind("127.0.0.1", https=https, warnings=warnings), 0, task="test", routes=routes)
        self.server.controller = controller
        self.server.start()
        test.addCleanup(self.server.stop)
        self.port = self.server.server_address[1]

    def request(self, method: str, path: str, *, host: str | None = None, body: dict | None = None,
                headers: dict | None = None) -> tuple[int, dict | bytes, str]:
        """Send one request. `host` is the Host header (default 127.0.0.1:<port>); a POST is sent as JSON.
        Returns the status code, the body (parsed when it's JSON) and its content type."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            sent = {"Host": host or f"127.0.0.1:{self.port}", **(headers or {})}
            data = None
            if method == "POST":
                data = json.dumps(body or {}).encode()
                sent.setdefault("Content-Type", "application/json")
            connection.request(method, path, body=data, headers=sent)
            response = connection.getresponse()
            raw = response.read()
            self.headers = response.headers
            kind = response.getheader("Content-Type") or ""
            return response.status, json.loads(raw) if kind == "application/json" else raw, kind
        finally:
            connection.close()


class CallerTest(unittest.TestCase):
    """Who may reach the page: its own addresses only, commands only from the page itself, behind
    HTTPS a Tailscale login unless the request is this Mac's own, and on 127.0.0.1 nothing through
    Tailscale Serve or Funnel. A refused request closes its connection, so a body the server didn't
    read is never parsed as more requests, which would skip the Host, Origin and login checks."""

    def setUp(self) -> None:
        self.controller = StubController()
        self.http = Server(self, self.controller)

    def embedded(self, command: str, last: bool = False) -> bytes:
        """A whole request from this Mac's own page, as it would pass every check."""
        body = json.dumps({"command": command}).encode()
        return (f"POST /command HTTP/1.1\r\nHost: 127.0.0.1:{self.http.port}\r\nContent-Type: application/json\r\n"
                f"X-SO101-Client: attacker\r\nContent-Length: {len(body)}\r\n"
                + ("Connection: close\r\n" if last else "") + "\r\n").encode() + body

    def raw(self, request: bytes) -> list[bytes]:
        """Send `request` on one connection and read until the server closes it. Returns the status
        lines of every response it sent."""
        with socket.create_connection(("127.0.0.1", self.http.port), timeout=3) as connection:
            connection.sendall(request)
            received = b""
            while chunk := connection.recv(65536):  # a connection the server leaves open times out
                received += chunk
        return [line.split(b"\r\n", 1)[0] for line in received.split(b"HTTP/1.1 ")[1:]]

    def check(self, http: Server, method: str, path: str, code: int, error: str | None = None, *,
              host: str | None = None, headers: dict | None = None, body: dict | None = None) -> dict | bytes:
        """Send one request and assert its answer: `code`, an error that names `error`, and that only
        an accepted command reached the controller. Returns the body."""
        before = len(self.controller.commands)
        if method == "POST" and body is None:
            body = {"command": "pause"}
        got, reply, _ = http.request(method, path, host=host, headers=headers, body=body)
        label = f"{method} {path} Host: {host} {headers} {body}"
        self.assertEqual(got, code, label)
        if error is not None:
            self.assertIn(error, reply["error"], label)
        self.assertEqual(len(self.controller.commands), before + (method == "POST" and code == 200),
                         f"{label}: only an accepted command reaches the controller")
        return reply

    def test_each_caller_gets_the_answer_its_host_origin_route_and_login_allow(self) -> None:
        plain, https = self.http, Server(self, self.controller, https=TS_NAME)

        # A site that points its own name at this address (DNS rebinding) sends its name as the Host.
        self.check(plain, "GET", "/status", 403, "only to its own addresses", host=f"evil.example:{plain.port}")
        self.check(plain, "POST", "/command", 403, host=f"evil.example:{plain.port}")
        for host in (f"127.0.0.1:{plain.port}", f"localhost:{plain.port}"):
            self.assertEqual(self.check(plain, "GET", "/status", 200, host=host)["state"], "resting")
        self.assertEqual(plain.headers["Content-Security-Policy"], "frame-ancestors 'none'")
        self.assertEqual(plain.headers["X-Frame-Options"], "DENY")
        self.assertEqual(plain.headers["Server"].strip(), "", "the response names no server software")

        # A command from another site, or a form or plain-text POST, which another site may send without asking.
        self.check(plain, "POST", "/command", 403, "only from the control page", headers={"Origin": "http://evil.example"})
        self.check(plain, "POST", "/command", 403, headers={"Content-Type": "text/plain"})
        self.check(plain, "POST", "/command", 200, headers={"Origin": f"http://127.0.0.1:{plain.port}"})
        self.assertEqual(self.controller.commands[-1]["who"], "127.0.0.1", "off HTTPS the caller is named by its address")
        # The reset sheet sends its attempt, and nothing else unknown.
        self.check(plain, "POST", "/command", 200, body={"command": "reset", "outcome": "failed", "for_attempt": 3})
        self.assertEqual(self.controller.commands[-1]["for_attempt"], 3)
        self.check(plain, "POST", "/command", 400, body={"command": "reset", "attempt": 3})

        # On 127.0.0.1 without HTTPS, a request Tailscale Serve or Funnel passed on is refused, whatever Host it names.
        local = f"localhost:{plain.port}"
        for headers in ({"X-Forwarded-For": "100.64.0.2"}, {"X-Forwarded-Host": TS_NAME},
                        {"Tailscale-Funnel-Request": "?1"},
                        {"X-Forwarded-For": "100.64.0.2", "Tailscale-User-Login": LOGIN}):
            self.check(plain, "GET", "/status", 403, 'bind = "tailscale"', host=local, headers=headers)
            self.check(plain, "POST", "/command", 403, host=local, headers={**headers, "Origin": f"http://{local}"})

        # Behind HTTPS, the tailnet name without a login is refused, naming Funnel.
        for path in ("/", "/status", "/events"):
            self.check(https, "GET", path, 403, "Funnel", host=TS_NAME)
        self.check(https, "POST", "/command", 403, "Funnel", host=TS_NAME, headers={"Origin": f"https://{TS_NAME}"})
        # Serve passes the Host on as sent, so a Funnel caller could name localhost. Serve's own
        # headers still show the request came through it.
        local = f"localhost:{https.port}"
        for headers in ({"X-Forwarded-For": "203.0.113.9"}, {"X-Forwarded-Host": local},
                        {"Tailscale-Funnel-Request": "?1"}):
            self.check(https, "GET", "/status", 403, "Funnel", host=local, headers=headers)
        # Funnel traffic is refused whatever else it carries.
        self.check(https, "GET", "/status", 403, host=TS_NAME,
                   headers={"Tailscale-User-Login": LOGIN, "Tailscale-Funnel-Request": "?1"})
        # The tailnet name with a login is served and names the caller.
        login = {"Tailscale-User-Login": LOGIN, "X-Forwarded-For": "100.64.0.2"}
        self.assertIn(b"SO-101 control", self.check(https, "GET", "/", 200, host=TS_NAME, headers=login))
        self.check(https, "POST", "/command", 200, host=TS_NAME, headers={**login, "Origin": f"https://{TS_NAME}"})
        self.assertEqual(self.controller.commands[-1]["who"], LOGIN)
        # This Mac's browser needs no login.
        for host in (f"127.0.0.1:{https.port}", f"localhost:{https.port}"):
            self.assertEqual(self.check(https, "GET", "/status", 200, host=host)["state"], "resting")
            self.check(https, "POST", "/command", 200, host=host, headers={"Origin": f"http://{host}"})
            self.assertEqual(self.controller.commands[-1]["who"], "this Mac")

        # A number JSON can't hold, and a POST with any Transfer-Encoding: the server never decodes
        # chunks, so it would read the body by Content-Length, or not at all, and parse what's left as
        # the next request on the connection.
        accepted = len(self.controller.commands)
        for body in (b'{"command":"reset","for_attempt":1e999}', b'{"command":"reset","for_attempt":Infinity}'):
            head = (f"POST /command HTTP/1.1\r\nHost: 127.0.0.1:{plain.port}\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n\r\n").encode()
            [status] = self.raw(head + body)
            self.assertTrue(status.startswith(b"400"), status)
        smuggled = self.embedded("resume", last=True)
        pause = b'{"command": "pause"}'
        chunked = f"{len(smuggled):x}\r\n".encode() + smuggled + b"\r\n0\r\n\r\n"
        for length, body in (("", chunked), (f"Content-Length: {len(pause)}\r\n", pause + smuggled), ("", smuggled)):
            with self.subTest(length=length):
                outer = (f"POST /command HTTP/1.1\r\nHost: 127.0.0.1:{plain.port}\r\nContent-Type: application/json\r\n"
                         f"Transfer-Encoding: chunked\r\n{length}\r\n").encode()
                [status] = self.raw(outer + body)
                self.assertTrue(status.startswith(b"400"), status)
        self.assertEqual(len(self.controller.commands), accepted, "no command reaches the controller")

    def test_a_cross_site_post_with_requests_in_its_body_gets_one_refusal(self) -> None:
        # Any website can make a browser send this: a text/plain POST needs no consent from the server.
        smuggled = self.embedded("take_control") + self.embedded("resume", last=True)
        for path, length, code in (("/command", len(smuggled), b"403"), ("/nowhere", len(smuggled), b"404"),
                                   ("/command", 99999, b"403")):
            with self.subTest(path=path, length=length):
                outer = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{self.http.port}\r\nOrigin: http://evil.example\r\n"
                         f"Content-Type: text/plain\r\nContent-Length: {length}\r\n\r\n").encode()
                [status] = self.raw(outer + smuggled)
                self.assertTrue(status.startswith(code), status)
        # The same body with a length the server won't read, from this Mac's own page.
        outer = (f"POST /command HTTP/1.1\r\nHost: 127.0.0.1:{self.http.port}\r\nContent-Type: application/json\r\n"
                 f"Content-Length: 99999\r\n\r\n").encode()
        [status] = self.raw(outer + smuggled)
        self.assertTrue(status.startswith(b"400"), status)
        self.assertEqual(self.controller.commands, [], "no command reaches the controller")

    def test_a_get_with_a_body_gets_one_refusal(self) -> None:
        # A GET's body is never read, so a GET that says it has one is refused rather than leave the
        # body to be parsed as the next request.
        smuggled = self.embedded("resume", last=True)
        for framing in (f"Content-Length: {len(smuggled)}", "Transfer-Encoding: chunked"):
            with self.subTest(framing=framing):
                outer = f"GET /status HTTP/1.1\r\nHost: 127.0.0.1:{self.http.port}\r\n{framing}\r\n\r\n".encode()
                [status] = self.raw(outer + smuggled)
                self.assertTrue(status.startswith(b"400"), status)
        self.assertEqual(self.controller.commands, [], "no command reaches the controller")


class UnsafeRouteTest(unittest.TestCase):
    """While a Tailscale route forwards raw connections to the port, or offers it through Funnel
    behind HTTPS (`route_warning`, checked every 5 s), every request is refused, even one that looks
    like this Mac's own browser. Even one with a login: a raw TCP forward passes on any header."""

    ROUTE = ("Tailscale Serve forwards TCP port 2222 straight to the control page's port 8101. Until it's gone, the "
             "control page refuses every request. Remove it with `tailscale serve --tcp=2222 off`.")

    def flag(self, http: Server, warning: str | None) -> str:
        """Report `warning` as the routes' latest check. Returns what it printed."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            http.server.routes_checked(warning)
        return printed.getvalue()

    def test_behind_https_nothing_is_served_until_it_clears(self) -> None:
        controller = StubController()
        http = Server(self, controller, https=TS_NAME, warnings=("Started behind HTTPS.",))
        local = {"Origin": f"http://127.0.0.1:{http.port}"}
        login = {"Tailscale-User-Login": LOGIN, "X-Forwarded-For": "100.64.0.2"}
        self.assertIn("TCP port 2222", self.flag(http, self.ROUTE))
        self.assertEqual(self.flag(http, self.ROUTE), "", "the warning is printed once")

        code, body, _ = http.request("GET", "/status")
        self.assertEqual(code, 403)
        self.assertIn("TCP port 2222", body["error"])
        self.assertEqual(http.request("POST", "/command", body={"command": "pause"}, headers=local)[0], 403)
        code, body, _ = http.request("GET", "/status", headers={"Tailscale-User-Login": LOGIN})
        self.assertEqual(code, 403, "a login sent over the raw forward proves nothing")
        self.assertEqual(controller.commands, [])

        self.assertIn("gone", self.flag(http, None))
        code, body, _ = http.request("GET", "/status")
        self.assertEqual((code, body["server_warning"]), (200, "Started behind HTTPS."))
        self.assertEqual(http.request("POST", "/command", body={"command": "pause"}, headers=local)[0], 200)
        self.assertEqual(http.request("GET", "/status", host=TS_NAME, headers=login)[0], 200)

        with self.subTest("on 127.0.0.1, this Mac's browser is refused until it clears"):
            http = Server(self, StubController())
            self.flag(http, self.ROUTE)
            for headers in ({}, {"Tailscale-User-Login": LOGIN}):
                code, body, _ = http.request("GET", "/", host=f"localhost:{http.port}", headers=headers)
                self.assertEqual(code, 403, headers)
                self.assertIn("TCP port 2222", body["error"])
            self.flag(http, None)
            code, body, _ = http.request("GET", "/status")
            self.assertEqual((code, body["server_warning"]), (200, None))

    def watch(self, https: str | None = None, reads: bool = True) -> tuple[Server, list[str | None], list[int]]:
        """A started server whose route watch asks every 10 ms, as `serve` does, with the Tailscale CLI
        faked, printing nothing at first unless it `reads`. Returns the server; a list whose one item
        is what the CLI prints for `serve status --json`, None for a CLI that printed nothing (it timed
        out, say); and the checks so far."""
        answer: list[str | None] = [serve_status({}) if reads else None]
        checks: list[int] = []

        def tailscale(*args: str):
            checks.append(1)
            return answer[0]

        for patch in (mock.patch.object(server, "ROUTES_EVERY_S", 0.01), mock.patch.object(server, "_tailscale", tailscale)):
            patch.start()
            self.addCleanup(patch.stop)
        bind = Bind("127.0.0.1", https=https)
        return Server(self, StubController(), https=https, routes=lambda: server.route_warning(bind, 8101)), answer, checks

    def checked(self, checks: list[int]) -> None:
        """Wait until the route watch has finished at least two more checks."""
        deadline, start = time.monotonic() + 5, len(checks)
        while len(checks) < start + 3:  # the third has started, so the two before it have finished
            if time.monotonic() > deadline:
                self.fail("the route watch stopped checking")
            time.sleep(0.005)

    def test_the_route_watch_refuses_until_a_check_reads_clean_and_survives_a_closed_terminal(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(server, "_tailscale_cli", return_value=None):
            http = self.watch(reads=False)[0]
            self.assertEqual(http.request("GET", "/status")[0], 200, "no CLI, no route")
            http.server.stop()  # so only the next server's watch asks the faked CLI

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed), \
                mock.patch.object(server, "_tailscale_cli", return_value="/usr/bin/tailscale"):
            http, answer, checks = self.watch(reads=False)
            code, body, _ = http.request("GET", "/status")
            self.assertEqual(code, 403, "a TCP forward may exist")
            self.assertIn("can't tell", body["error"])
            answer[0] = serve_status({})
            self.checked(checks)
            self.assertEqual(http.request("GET", "/status")[0], 200, "a status that reads clean lifts it")

            answer[0] = serve_status({}, tcp={"2222": {"TCPForward": "127.0.0.1:8101"}})
            self.checked(checks)
            self.assertEqual(http.request("GET", "/status")[0], 403)
            # The CLI timed out or printed nothing, tailscaled didn't answer, or the output isn't JSON.
            for unreadable in (None, "", "not json", "Failed to connect to local Tailscale daemon"):
                answer[0] = unreadable
                self.checked(checks)
                code, body, _ = http.request("GET", "/status")
                self.assertEqual(code, 403, f"{unreadable!r}: the route may still be there")
                self.assertIn("TCP port 2222", body["error"])
            answer[0] = serve_status({})
            self.checked(checks)
            self.assertEqual(http.request("GET", "/status")[0], 200, "a status that reads clean lifts it")
        self.assertEqual(printed.getvalue().count("TCP port 2222"), 1, printed.getvalue())
        self.assertEqual(printed.getvalue().count("The Tailscale route is gone"), 1, printed.getvalue())

        with contextlib.redirect_stdout(DeadStream()):  # every print fails, as after the terminal closes
            answer[0] = serve_status({}, tcp={"2222": {"TCPForward": "127.0.0.1:8101"}})
            self.checked(checks)
            self.assertEqual(http.request("GET", "/status")[0], 403)
            answer[0] = serve_status({})
            self.checked(checks)
            self.assertEqual(http.request("GET", "/status")[0], 200, "the watch goes on after a print fails")


def serve_config(web: dict, funnel: dict | None = None, tcp: dict | None = None) -> dict:
    """One Serve config, as Tailscale prints it: each Web route's port served over HTTPS unless
    `tcp` says otherwise."""
    return {"TCP": {**{hostport.rsplit(":", 1)[1]: {"HTTPS": True} for hostport in web}, **(tcp or {})},
            "Web": {hostport: {"Handlers": handlers} for hostport, handlers in web.items()},
            **({"AllowFunnel": funnel} if funnel else {})}


def serve_status(web: dict, funnel: dict | None = None, tcp: dict | None = None) -> str:
    """`tailscale serve status --json` output, as Tailscale prints it."""
    return json.dumps(serve_config(web, funnel, tcp))


class FakedTailscale(unittest.TestCase):
    """`resolve_bind`, with the Tailscale CLI's output faked."""

    PORT = 8101
    TS_IP = "100.101.102.103"
    WEB = {f"{TS_NAME}:443": {"/": {"Proxy": "http://127.0.0.1:8101"}}}

    def resolve(self, bind: str, *, ip: str = "", serve: str = "") -> server.Bind:
        """resolve_bind for `bind`, with the Tailscale CLI printing `ip` for `ip -4` and `serve` for
        `serve status --json`."""
        outputs = {"ip": ip, "serve": serve, "status": json.dumps({"Self": {"DNSName": f"{TS_NAME}."}})}

        def tailscale(*args: str):
            return outputs[args[0]]

        with mock.patch.object(server, "_tailscale", tailscale), \
                mock.patch.object(server.subprocess, "run", side_effect=AssertionError("only the Tailscale CLI runs")):
            return server.resolve_bind(control(bind=bind, port=self.PORT))


class ServeStatusTest(FakedTailscale):
    """What `resolve_bind` makes of `tailscale serve status --json` output, as Tailscale prints it,
    with bind = "tailscale" and Tailscale running."""

    def https(self, out: str) -> server.Bind:
        return self.resolve("tailscale", ip=self.TS_IP, serve=out)

    def test_funnel_or_a_tcp_forward_on_any_route_to_the_port_is_reported(self) -> None:
        out = serve_status({f"{TS_NAME}:443": {"/": {"Proxy": "http://127.0.0.1:8101"}},
                            f"{TS_NAME}:8443": {"/": {"Proxy": "http://127.0.0.1:8101"}}},
                           funnel={f"{TS_NAME}:8443": True})
        self.assertRefused(out, f"{TS_NAME}:8443", "Funnel")
        out = serve_status({f"{TS_NAME}:443": {"/": {"Proxy": "http://127.0.0.1:8101"}},
                            f"{TS_NAME}:10000": {"/robot": {"Proxy": "http://127.0.0.1:8101/"}}},
                           funnel={f"{TS_NAME}:10000": True})
        self.assertRefused(out, f"{TS_NAME}:10000", "Funnel")  # a route on another path counts too
        # `tailscale funnel 8101` run in the foreground keeps its route under Foreground.
        background = serve_config({f"{TS_NAME}:443": {"/": {"Proxy": "http://127.0.0.1:8101"}}})
        session = serve_config({f"{TS_NAME}:8443": {"/": {"Proxy": "http://127.0.0.1:8101"}}},
                               funnel={f"{TS_NAME}:8443": True})
        out = json.dumps({**background, "Foreground": {"a1b2c3": session}})
        self.assertRefused(out, f"{TS_NAME}:8443", "Funnel")
        # A TCP forward to the port passes raw connections on, with no Serve headers to tell them from
        # this Mac's own browser.
        out = serve_status({f"{TS_NAME}:443": {"/": {"Proxy": "http://127.0.0.1:8101"}}},
                           tcp={"2222": {"TCPForward": "localhost:8101"}})
        self.assertIn("tailscale serve --tcp=2222 off", self.assertRefused(out, "2222", "TCP"))

    def assertRefused(self, out: str, route: str, kind: str) -> str:
        """No HTTPS: the Tailscale address, with a warning that names `route` and `kind` of route.
        Returns the warning."""
        bind = self.https(out)
        self.assertEqual((bind.address, bind.https), (self.TS_IP, None))
        warning = " ".join(bind.warnings)
        self.assertTrue(warning)
        self.assertIn(route, warning)
        self.assertIn(kind, warning)
        return warning


class ResolveBindTest(FakedTailscale):
    """Where the page listens, with the Tailscale CLI's output faked."""

    def test_on_127_0_0_1_a_tcp_forward_to_the_port_stops_serve(self) -> None:
        # Its connections carry no Serve headers, so they would pass for this Mac's own browser.
        terminated = {"10000": {"TCPForward": "127.0.0.1:8101", "TerminateTLS": TS_NAME}}
        for funnel, off in ((None, "tailscale serve --tls-terminated-tcp=10000 off"),
                            ({f"{TS_NAME}:10000": True}, "tailscale funnel --tls-terminated-tcp=10000 off")):
            with self.subTest(funnel=funnel):
                with self.assertRaises(ValueError) as refused:
                    self.resolve("127.0.0.1", ip=self.TS_IP, serve=serve_status({}, funnel=funnel, tcp=terminated))
                message = str(refused.exception)
                self.assertIn("TCP port 10000", message)
                self.assertIn(f"`{off}`", message)
                self.assertNotIn("Tailscale tailscale", message)

    def test_on_127_0_0_1_a_web_route_only_warns(self) -> None:
        # Serve marks each request it passes on, so the page refuses them one by one.
        bind = self.resolve("127.0.0.1", ip=self.TS_IP, serve=serve_status(self.WEB, funnel={f"{TS_NAME}:443": True}))
        self.assertEqual((bind.address, bind.https), ("127.0.0.1", None))
        [warning] = bind.warnings
        self.assertIn(f"Funnel offers {TS_NAME}:443", warning)
        self.assertEqual(self.resolve("127.0.0.1", ip=self.TS_IP), server.Bind("127.0.0.1"), "no route, no warning")
        self.assertEqual(self.resolve("tailscale"), server.Bind("127.0.0.1", warnings=(server.NO_TAILSCALE,)),
                         "Tailscale stopped: the page stays on this computer, and it says so")

    def test_routes_checked_while_serve_runs(self) -> None:
        def check(bind: server.Bind, serve: str) -> str | None:
            with mock.patch.object(server, "_tailscale", lambda *args: serve):
                return server.route_warning(bind, self.PORT)

        tcp = serve_status({}, tcp={"2222": {"TCPForward": "127.0.0.1:8101"}})
        funnel = serve_status(self.WEB, funnel={f"{TS_NAME}:443": True})
        loopback, https = server.Bind("127.0.0.1"), server.Bind("127.0.0.1", https=TS_NAME)
        for bind in (loopback, https):
            self.assertIn("`tailscale serve --tcp=2222 off`", check(bind, tcp))
            self.assertIsNone(check(bind, serve_status(self.WEB)))
        self.assertIn("Funnel", check(https, funnel), "behind HTTPS, Funnel on the port is unsafe")
        self.assertIsNone(check(loopback, funnel), "off HTTPS, each request through Funnel is refused already")
        self.assertIsNone(check(server.Bind(self.TS_IP), tcp), "a route to localhost never reaches the Tailscale address")


if __name__ == "__main__":
    unittest.main()

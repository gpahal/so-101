"""The web server every `so101` page runs on: the standard library's ThreadingHTTPServer, in a daemon
thread, with the checks that keep a page off the public internet. `control/server.py` (the control
page, `so101 serve`) and `web/review.py` (the reviewer, `so101 review`) subclass `PageServer` and
`Handler` and add their routes; everything below holds for both.

`resolve_bind` decides where a page listens, once, before anything connects. With bind = "tailscale"
it listens only on this computer's Tailscale address, found with the `tailscale` CLI: the one on the
PATH, else the Mac app's. No CLI, or a CLI that gives no address, means Tailscale is stopped: the
page falls back to 127.0.0.1 and warns. It never listens on every interface (0.0.0.0): `bind` takes
only "tailscale" or "127.0.0.1", and Tailscale is the page's encryption and access control.

When `tailscale serve` already offers the port over HTTPS (`serve_routes`), the server listens
on 127.0.0.1 behind it, and the page's address is https://<name>.<tailnet>.ts.net/. HTTPS lets the
control page keep a phone's screen awake (the Screen Wake Lock API needs it). Serve passes each
request on with the caller's Tailscale login in `Tailscale-User-Login`, for a device signed in as a
user, and strips any copy the caller sent. Funnel's public traffic never has a login, and Serve marks
it with `Tailscale-Funnel-Request`. So in HTTPS mode a request that came through Serve, told by its
Host (the ts.net name) or by the X-Forwarded-* headers Serve adds, is refused without a login, even
when Funnel was turned on after the page started. A request to 127.0.0.1:<port> or localhost:<port>
that didn't come through Serve can only come from this Mac, which needs no login. That holds only
while no Serve or Funnel route forwards raw TCP to the port (`--tcp` or `--tls-terminated-tcp`):
those connections arrive with no Serve headers, like this Mac's own. So a page never goes behind
HTTPS while one does, nor while Funnel offers any route to the port. Off HTTPS, on 127.0.0.1, the
page is for this Mac's browser only, and a request Serve or Funnel passed on is refused; the page
won't start there at all while a TCP forward to the port exists. On the Tailscale address, a Funnel
TCP forward to the port at any address but this Mac's loopback could reach the page from the public
internet, so it is unsafe there. Routes can change while a page runs, so they are checked again
every ROUTES_EVERY_S (1 s) on a thread of the server's own (`_watch_routes`, calling `route_warning`), never a request's, so
no request, the control page's Pause included, waits on the Tailscale CLI. While an unsafe route
exists, every request is refused with a message naming the route, until it's gone. A check that
can't read Tailscale's status (the CLI timed out, or tailscaled didn't answer) keeps the last
answer. The first check runs before the first request is served, and on 127.0.0.1 a first check that
can't read the status, with a Tailscale CLI installed, refuses every request until a check reads it.

Every request needs a Host header naming an address the server answers to: the address and port it
listens on; this computer's MagicDNS name (short or full) with the port when it listens on its
Tailscale address; localhost:port when it listens on 127.0.0.1; and the HTTPS name behind Tailscale
Serve. So a site that points its own name at this address (DNS rebinding) can't read a page's data
or send it anything. POST also needs `Content-Type: application/json` and, when the browser sends an
Origin, one naming an address the server answers to. A web page on another site can't send that
request without the server's consent, which it never gives, so only the page itself can press its
buttons. Every response with an error status closes its connection, so the body of a refused
request, which the server may not have read, is never parsed as further requests. Any request with a
Transfer-Encoding, and a GET that says it has a body (a Content-Length other than 0), is refused with
400 for the same reason: the server reads a body only by Content-Length, and never a GET's. A
connection that sends nothing for 10 s is closed. Every response forbids framing the page and names
no server software. A route that raises before it has sent anything is answered 500 with the error's
name (`Handler._route`), and the error is printed in the terminal, so a page never sees a dropped
connection where it expects a reply.

The messages name the page and the command that serves it through a `Site`, so the reviewer's say
`so101 review` where the control page's say `so101 serve`.
"""

import contextlib
import ipaddress
import json
import math
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

from so_101 import svtshim

TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")
LOOPBACK = "127.0.0.1"
# Funnel puts a port on the public internet. The page refuses Funnel's requests, which carry no
# Tailscale login, but it still never runs behind a route that Funnel offers.
FUNNEL = ("{route}, so the {page} stays off HTTPS and listens on the Tailscale address instead: {guards} never "
          "go behind a public route. Turn Funnel off for it to use HTTPS (`tailscale funnel status`).")
# A route that passes raw TCP connections to the page's port: they carry no Serve headers, so they
# would pass for this Mac's own browser.
TCP_FORWARD = ("{route}. Those connections carry no Tailscale login and look like this Mac's own browser, so the "
               "{page} stays off HTTPS and listens on the Tailscale address instead. Remove the route with "
               "`{off}` to use HTTPS.")
# The same route while the page would listen on 127.0.0.1: the command stops before anything connects.
TCP_ON_LOOPBACK = ("{route}. Those connections carry no Tailscale login and look like this Mac's own browser, so the "
                   "{page} won't listen on 127.0.0.1 while the route exists. Remove it with `{off}`{instead}, "
                   "then start `{command}` again.")
# A web route to the page's port while it listens on 127.0.0.1 without HTTPS: Serve marks each
# request it passes on, so each is refused (LOOPBACK_ONLY).
WEB_ON_LOOPBACK = ("{route}, but the {page} listens on 127.0.0.1 for this Mac's browser only, so it refuses "
                   "every request through that route. {fix}")
# A route found while the page runs (`route_warning`) that makes a request without a login unsafe.
ROUTE_APPEARED = ("{route}. Until it's gone, the {page} refuses every request, this Mac's own browser's "
                  "too. {fix}")
# The first check, with a Tailscale CLI installed, couldn't read the status: a TCP forward may exist.
ROUTES_UNREAD = ("The `tailscale` command is installed but gave no status to read, so the {page} can't tell "
                 "whether a route forwards raw TCP to its port. It refuses every request until a check reads the "
                 "status, every second. Start Tailscale if it's stopped, or see why `tailscale serve status --json` fails.")
# How often a page checks Tailscale's routes again, on the web server's own thread: a route made since
# the last check reaches the page for at most this long. Each check runs the Tailscale CLI.
ROUTES_EVERY_S = 1.0
# A request Serve or Funnel passed on, while the page listens on 127.0.0.1 without HTTPS.
LOOPBACK_ONLY = ("This request came through Tailscale Serve or Funnel, but the {page} listens on 127.0.0.1 for a "
                 "browser on this computer only (bind = \"127.0.0.1\" in so101.toml, or Tailscale wasn't running when "
                 "`{command}` started), so it refuses it. To use the page through Tailscale, set bind = \"tailscale\" "
                 "under [control] and restart `{command}`.")
# A request through Tailscale Serve without a login: Funnel's, or a tagged device's.
NO_LOGIN = ("This request came through Tailscale without a Tailscale login, as every request from the public internet "
            "through Tailscale Funnel does, so the {page} refuses it. Turn Funnel off for this port (`tailscale "
            "funnel status`), and open the page from a device signed in to your tailnet as a user: a tagged device "
            "has no login either.")
NO_TAILSCALE = ("Tailscale isn't running, so the {page} is on this computer only and no {remote} can "
                "reach it. Start Tailscale and restart `{command}` to allow {remotes}.")
# Where the Tailscale CLI is when it isn't on the PATH: inside the Mac app.
TAILSCALE_APP = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
MAX_BODY = 4096
LOGIN_HEADER = "Tailscale-User-Login"  # Serve's: the caller's Tailscale login
FUNNEL_HEADER = "Tailscale-Funnel-Request"  # Serve's: the request came from the public internet
# Headers Tailscale Serve adds to every request it passes on, and a browser never sends.
SERVE_HEADERS = ("X-Forwarded-For", "X-Forwarded-Host", FUNNEL_HEADER)
THIS_MAC = "this Mac"  # how a page names a request from this computer's own browser


@dataclass(frozen=True)
class Site:
    """How a page's messages name it: "control page"; the command that serves it, "so101 serve"; what
    its POSTs are, "Commands"; what never goes behind a public route, "the arm's controls"; who reaches
    it from elsewhere, "remote operator"; and its threads' name, "control-page"."""

    page: str
    command: str
    posts: str
    guards: str
    remote: str
    thread: str

    def say(self, template: str, **fields) -> str:
        return template.format(page=self.page, command=self.command, guards=self.guards, remote=self.remote,
                               remotes=f"{self.remote}s", **fields)


def _tailscale_cli() -> str | None:
    """The Tailscale CLI on this computer, the one on the PATH or else the Mac app's, or None."""
    cli = shutil.which("tailscale") or TAILSCALE_APP
    return cli if Path(cli).exists() else None


def _tailscale(*args: str) -> str | None:
    """What the Tailscale CLI prints for `args`, or None when there's no CLI or it fails to run."""
    if (cli := _tailscale_cli()) is None:
        return None
    try:
        return subprocess.run([cli, *args], capture_output=True, text=True, timeout=3,
                              env=svtshim.other_env()).stdout
    except (OSError, subprocess.SubprocessError):
        return None


Tailscale = Callable[..., str | None]  # what the Tailscale CLI prints for its arguments, as `_tailscale`


def tailscale_ip(tailscale: Tailscale | None = None) -> str | None:
    """This computer's Tailscale IPv4 address, or None when Tailscale isn't running or no CLI exists."""
    for word in ((tailscale or _tailscale)("ip", "-4") or "").split():
        try:
            if ipaddress.ip_address(word) in TAILSCALE_NET:
                return word
        except ValueError:
            continue
    return None


def tailscale_name(tailscale: Tailscale | None = None) -> str | None:
    """This computer's MagicDNS name, such as `studio.tail1234.ts.net`, from `tailscale status
    --json`, or None when Tailscale doesn't say."""
    try:
        name = json.loads((tailscale or _tailscale)("status", "--json") or "").get("Self", {}).get("DNSName")
    except (ValueError, AttributeError):
        return None
    return name.strip(".").lower() if isinstance(name, str) and name.strip(".") else None


@dataclass(frozen=True)
class Route:
    """A Tailscale Serve or Funnel route to the page's port, from `tailscale serve status --json`."""

    port: int  # the page's port
    where: str  # the TCP port a TCP route forwards, or the host:port a web route serves
    funnel: bool  # open to the public internet
    tcp: bool  # raw TCP connections, with no Serve headers; otherwise a web route, whose requests carry them
    site: Site  # the page it reaches, for the messages
    off: str = ""  # for a TCP route: the command that removes it
    loopback: bool = True  # it forwards to localhost or 127.0.0.1; a TCP route may name any address

    def __str__(self) -> str:
        public = ", open to the public internet" if self.funnel else ""
        if self.tcp:
            return (f"Tailscale {'Funnel' if self.funnel else 'Serve'} forwards TCP port {self.where} straight to the "
                    f"{self.site.page}'s port {self.port}{public}")
        if self.funnel:
            return f"Tailscale Funnel offers {self.where} to the public internet"
        return f"Tailscale Serve offers {self.where} to your tailnet"

    @property
    def fix(self) -> str:
        if self.tcp:
            return f"Remove it with `{self.off}`."
        if self.funnel:
            return "Turn Funnel off for it (`tailscale funnel status`)."
        return (f"To use the page through Tailscale, set bind = \"tailscale\" under [control] and restart "
                f"`{self.site.command}`.")


@dataclass(frozen=True)
class Bind:
    """Where a page listens (`resolve_bind`): its address; the host where `tailscale serve` offers it
    over HTTPS, or None; this computer's MagicDNS name when it listens on its Tailscale address, or
    None; and what the command prints and the page shows about that choice."""

    address: str
    https: str | None = None
    dns_name: str | None = None
    warnings: tuple[str, ...] = ()


def resolve_bind(bind: str, port: int, site: Site, tailscale: Tailscale | None = None) -> Bind:
    """Where the page on `port` listens for `bind`, so101.toml's `[control] bind`. Tailscale's routes
    to the port (`tailscale serve status --json`) are checked whatever `bind` says. `tailscale` runs
    the CLI (`_tailscale` by default).

    With bind = "tailscale": 127.0.0.1 behind HTTPS when `tailscale serve` offers the port over HTTPS
    and no route makes that unsafe (Funnel on any route to the port, or any TCP forward to it);
    otherwise the Tailscale address, with a warning for each unsafe route. When Tailscale isn't
    running, 127.0.0.1, with a warning.

    On 127.0.0.1 without HTTPS, a TCP forward to the port raises ValueError, naming the route and the
    command that removes it: its connections would pass for this Mac's own browser. A web route only
    warns, because Serve marks each request it passes on, and the page refuses those."""
    tailscale = tailscale or _tailscale
    # A CLI that's absent means no route can exist. A status that can't be read (a stopped tailscaled,
    # a timeout) is read as no routes here too, but not for long: `PageServer.start` checks again,
    # and on 127.0.0.1 refuses every request until a check reads it.
    found, routes = serve_routes(tailscale("serve", "status", "--json") or "", port, site) or (None, [])
    warnings: list[str] = []
    if bind == "tailscale":
        # A TCP forward to another address can't reach a page behind HTTPS, which listens on 127.0.0.1.
        unsafe = [route for route in routes if (route.tcp and route.loopback) or route.funnel]
        if found and not unsafe:
            return Bind(LOOPBACK, https=found)
        if address := tailscale_ip(tailscale):
            warnings = [site.say(TCP_FORWARD if route.tcp else FUNNEL, route=route, off=route.off) for route in unsafe]
            return Bind(address, dns_name=tailscale_name(tailscale), warnings=tuple(warnings))
        warnings = [site.say(NO_TAILSCALE)]
    if tcp := [route for route in routes if route.tcp and route.loopback]:
        instead = "" if bind == "tailscale" else ", or set bind = \"tailscale\" under [control] in so101.toml"
        raise ValueError(" ".join(site.say(TCP_ON_LOOPBACK, route=route, off=route.off, instead=instead)
                                  for route in tcp))
    warnings += [site.say(WEB_ON_LOOPBACK, route=route, fix=route.fix) for route in routes if not route.tcp]
    return Bind(LOOPBACK, warnings=tuple(warnings))


def route_warning(bind: Bind, port: int, site: Site, tailscale: Tailscale | None = None) -> str | None:
    """What makes a request unsafe now, from Tailscale's routes to `port`, or None. On 127.0.0.1: a TCP
    forward to the port there, whose connections pass for this Mac's own browser, and behind HTTPS
    also Funnel on any route to it. On the Tailscale address: a Funnel TCP forward to the port at an
    address other than the loopback, which may be this one, open to the public internet. Asked again
    every ROUTES_EVERY_S while the page runs, as routes can change without it.

    Raises RuntimeError when the CLI gave no status it could read (it's absent, it timed out,
    tailscaled didn't answer): "no routes" then would lift a refusal while the route may still be there."""
    if (status := serve_routes((tailscale or _tailscale)("serve", "status", "--json") or "", port, site)) is None:
        raise RuntimeError("Tailscale gave no `tailscale serve status --json` to read.")
    if bind.address == LOOPBACK:
        risky = [route for route in status[1] if (route.tcp and route.loopback) or (bind.https and route.funnel)]
    else:
        risky = [route for route in status[1] if route.tcp and route.funnel and not route.loopback]
    return " ".join(site.say(ROUTE_APPEARED, route=route, fix=route.fix) for route in risky) or None


def serve_routes(out: str, port: int, site: Site) -> tuple[str | None, list[Route]] | None:
    """From `tailscale serve status --json` output: the HTTPS host and port whose root proxies to
    localhost:`port` or 127.0.0.1:`port`, with port 443 left off, as a browser leaves it off, or None;
    and every route to the port. Background and foreground (`Foreground`) sessions count alike. A
    route whose port Serve offers over plain HTTP isn't HTTPS. None for output that isn't a JSON
    object, as Tailscale's status always is (`{}` with no routes): the status couldn't be read."""
    try:
        status = json.loads(out)
    except ValueError:
        return None
    if not isinstance(status, dict):
        return None
    # `tailscale serve` run in the foreground keeps its routes under Foreground, one config per session.
    configs = [status, *_dict(status.get("Foreground")).values()]
    found, routes = None, []
    for config in map(_dict, configs):
        public = _dict(config.get("AllowFunnel"))
        tcp = _dict(config.get("TCP"))
        for tcp_port, entry in tcp.items():
            entry = _dict(entry)
            # Any address counts: a forward to this Mac's Tailscale address reaches a page bound there.
            if not _proxies_to(entry.get("TCPForward"), port, any_host=True):
                continue
            funnel = any(hostport.rsplit(":", 1)[-1] == str(tcp_port) and allowed is True
                         for hostport, allowed in public.items())
            flag = "--tls-terminated-tcp" if entry.get("TerminateTLS") else "--tcp"
            routes.append(Route(port, str(tcp_port), funnel, tcp=True, site=site,
                                off=f"tailscale {'funnel' if funnel else 'serve'} {flag}={tcp_port} off",
                                loopback=_proxies_to(entry.get("TCPForward"), port)))
        for hostport, entry in _dict(config.get("Web")).items():
            handlers = _dict(_dict(entry).get("Handlers"))
            to_port = [path for path, handler in handlers.items() if _proxies_to(_dict(handler).get("Proxy"), port)]
            if not to_port:
                continue
            funnel = public.get(hostport) is True
            routes.append(Route(port, hostport.lower(), funnel, tcp=False, site=site))
            if (not funnel and "/" in to_port and found is None
                    and _dict(tcp.get(hostport.rsplit(":", 1)[-1])).get("HTTPS") is True):
                found = hostport.lower().removesuffix(":443")
    return found, routes


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


# Targets a forward on this Mac reaches its own loopback through: 0.0.0.0 and :: connect locally on macOS.
LOCAL_HOSTS = ("localhost", LOOPBACK, "0.0.0.0", "::", "::1")


def _proxies_to(proxy, port: int, any_host: bool = False) -> bool:
    """Whether a Serve handler's Proxy, such as `http://127.0.0.1:8101`, is this computer's `port`
    on its loopback, or with `any_host`, `port` at any address."""
    if not isinstance(proxy, str) or not proxy:
        return False
    if proxy.isdigit():
        return int(proxy) == port
    try:
        target = urlsplit(proxy if "://" in proxy else f"http://{proxy}")
        return (any_host or target.hostname in LOCAL_HOSTS) and target.port == port
    except ValueError:
        return False


def _finite(text: str) -> float:
    """A JSON number that a float holds. `json.loads` would take NaN and Infinity, which aren't JSON,
    and turn 1e999 into inf, which a reply would then send as Infinity, and no page could parse it.
    Both hooks: `parse_constant` gets the names, `parse_float` the numbers."""
    if not math.isfinite(value := float(text)):
        raise ValueError(f"{text} isn't a JSON number a float holds")
    return value


class PageServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, bind: Bind, port: int, handler: type["Handler"], *, site: Site,
                 routes: Callable[[], str | None]) -> None:
        """Binds now, so a port in use fails at once. `bind` is where to listen, from `resolve_bind`,
        and `warning` joins its warnings. `routes()` returns what makes a request without a login unsafe
        (`route_warning`), asked before the first request and then every ROUTES_EVERY_S on a thread of
        its own."""
        super().__init__((bind.address, port), handler)
        self.site = site
        self.warning = " ".join(bind.warnings) or None
        self.dns_name = bind.dns_name
        self.https = bind.https
        # The Host headers it answers to: its address, `localhost` on 127.0.0.1, and the MagicDNS name
        # in full and as its first label, each with the port; and the HTTPS name as the browser names
        # it, which Tailscale Serve passes on.
        host, bound_port = self.server_address[:2]
        names = [host, *(["localhost"] if host == LOOPBACK else [])]
        if self.dns_name:
            names += [self.dns_name, self.dns_name.split(".", 1)[0]]
        self.hosts = frozenset({f"{name}:{bound_port}" for name in names} | ({self.https} if self.https else set()))
        self.routes = routes
        # While set, the route that makes a request's origin unknowable: every request is refused with it.
        self.route_warning: str | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    @property
    def url(self) -> str:
        if self.https:
            return f"https://{self.https}/"
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"

    @property
    def name_url(self) -> str | None:
        """The page's address by MagicDNS name, or None without one."""
        return f"http://{self.dns_name}:{self.server_address[1]}/" if self.dns_name else None

    def start(self) -> None:
        self._first_route_check()
        self._thread = threading.Thread(target=self.serve_forever, name=self.site.thread, daemon=True)
        self._thread.start()
        threading.Thread(target=self._watch_routes, name=f"{self.site.thread}-routes", daemon=True).start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self.shutdown()
        self.server_close()

    def _first_route_check(self) -> None:
        """Check the routes once before the first request is served. `resolve_bind` read an unreadable
        status as no routes, so with a CLI installed the page refuses every request until a check
        reads it. With no CLI there's no Tailscale, and no route can exist."""
        try:
            warning = self.routes()
        except Exception:
            warning = self.site.say(ROUTES_UNREAD) if _tailscale_cli() is not None else None
        self.routes_checked(warning)

    def _watch_routes(self) -> None:
        while not self._stopping.wait(ROUTES_EVERY_S):
            try:
                warning = self.routes()
            except Exception:  # a check that fails, or can't read the status, keeps the last answer
                continue
            self.routes_checked(warning)

    def routes_checked(self, warning: str | None) -> None:
        """The routes' latest check found `warning`, or nothing unsafe. A new warning is printed once,
        and while it stands every request is refused with it, so the page shows it. A request with a
        Tailscale login is refused too: a raw TCP forward passes on any header its caller sends, so
        a login proves nothing while one exists. A message is printed when it clears."""
        if warning == self.route_warning:
            return
        previous, self.route_warning = self.route_warning, warning
        cleared = (f"Tailscale's status reads clean: the {self.site.page} serves requests."
                   if previous == self.site.say(ROUTES_UNREAD)
                   else f"The Tailscale route is gone: the {self.site.page} serves requests again.")
        # On the watch's own thread: a print that fails, as after the terminal closes, must not end it.
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            print(warning or cleared, flush=True)

    def handle_error(self, request, client_address) -> None:
        """A caller that hung up mid-reply, as a phone does when it locks during a poll, ends only its
        own request, quietly: printed, its traceback would land among the terminal's messages. Any
        other error is printed."""
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    """Checks every request, then hands it to `get` or `post`, which a page's handler defines. A POST
    reaches `post` only for a path in POSTS, from the page itself."""

    server: PageServer
    protocol_version = "HTTP/1.1"
    timeout = 10  # seconds a connection may sit silent, so a stalled client can't hold a thread for good
    server_version = ""  # with sys_version, leaves the Server header empty
    sys_version = ""
    POSTS: tuple[str, ...] = ()  # the paths that take a POST

    def log_message(self, format: str, *args) -> None:
        pass  # a page's polls would bury the terminal's messages

    def get(self, url: SplitResult) -> None:
        self._json(HTTPStatus.NOT_FOUND, {"error": f"No {url.path} here."})

    def post(self, url: SplitResult) -> None:
        self._json(HTTPStatus.NOT_FOUND, {"error": f"No {self.path} here."})

    def _refuse_framing(self) -> bool:
        """Answer 400 to a request whose body the server wouldn't read in full, and return True: any
        Transfer-Encoding, as it reads only by Content-Length, and a GET with a body, which it never
        reads. What's left would be parsed as the next request on the connection, skipping every check."""
        length = self.headers.get("Content-Length")
        if self.headers.get("Transfer-Encoding") is not None:
            error = "Send a Content-Length: Transfer-Encoding isn't accepted."
        elif self.command == "GET" and length is not None and length.strip() != "0":
            error = "A GET carries no body."
        else:
            return False
        self._json(HTTPStatus.BAD_REQUEST, {"error": error})
        return True

    def do_GET(self) -> None:
        self._sent = False
        if self._refuse_framing():
            return
        if (error := self._refuse_caller()) is not None:
            self._json(HTTPStatus.FORBIDDEN, {"error": error})
            return
        self._route(self.get, urlsplit(self.path))

    def do_POST(self) -> None:
        self._sent = False
        if self._refuse_framing():
            return
        url = urlsplit(self.path)
        if url.path not in self.POSTS:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"No {self.path} here."})
            return
        if (error := self._refuse_caller() or self._refuse_cross_site()) is not None:
            self._json(HTTPStatus.FORBIDDEN, {"error": error})
            return
        self._route(self.post, url)

    def _route(self, handle: Callable[[SplitResult], None], url: SplitResult) -> None:
        """Run a page's `get` or `post` after every check passed. An error it didn't answer itself is a 500
        naming the error, when nothing was sent yet, and its message goes to the terminal; once a response
        has started, the error closes the connection, since a second response can't follow."""
        try:
            handle(url)
        except Exception as error:
            if getattr(self, "_sent", True):  # `handle_error` deals with it, as with any error mid-reply
                raise
            with contextlib.suppress(OSError, ValueError):  # stderr gone or closed: the reply still goes
                print(f"The {self.server.site.page} couldn't answer {self.command} {url.path}: "
                      f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                       {"error": f"The {self.server.site.page} couldn't answer: {type(error).__name__}. "
                                 f"The terminal running `{self.server.site.command}` says more."})

    def _read_json(self) -> dict:
        """The POST's body, a JSON object of at most MAX_BODY bytes, read by Content-Length. Raises
        ValueError for anything else, including a number a float can't hold."""
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= MAX_BODY:
            raise ValueError("negative or too long")
        body = json.loads(self.rfile.read(length) or b"{}", parse_constant=_finite, parse_float=_finite)
        if not isinstance(body, dict):
            raise ValueError("not an object")
        return body

    def _refuse_caller(self) -> str | None:
        """Refuse a request whose Host isn't this server's own address (a rebound DNS name), one from
        Funnel, in HTTPS mode one through Tailscale Serve without a Tailscale login, and on 127.0.0.1
        without HTTPS any request through Serve. While an unsafe route exists (`routes_checked`),
        refuse every request."""
        site = self.server.site
        if self._through_serve() and not self.server.https and self.server.server_address[0] == LOOPBACK:
            return site.say(LOOPBACK_ONLY)
        if self.headers.get(FUNNEL_HEADER) is not None:
            return site.say(NO_LOGIN)
        if route := self.server.route_warning:
            # Not even with a login: a raw TCP forward passes on whatever headers its caller sends.
            return route
        host = (self.headers.get("Host") or "").lower()
        if host not in self.server.hosts:  # names no address: the caller may be anyone who can reach the port
            return f"This server answers only to its own addresses. Open the {site.page} at the one `{site.command}` printed."
        if self.server.https and not self._login() and self._through_serve():
            return site.say(NO_LOGIN)
        return None

    def _through_serve(self) -> bool:
        """Whether Tailscale Serve passed this request on: its Host is the HTTPS name, or it carries a
        header Serve adds. Serve passes the Host on as the caller sent it, so the headers count too."""
        host = (self.headers.get("Host") or "").lower()
        return host == self.server.https or any(self.headers.get(name) is not None for name in SERVE_HEADERS)

    def _login(self) -> str:
        """The caller's Tailscale login from Serve's header, or ""."""
        return (self.headers.get(LOGIN_HEADER) or "").strip()

    def _who(self) -> str:
        """How a page names the caller of a request `_refuse_caller` let through: behind HTTPS its
        Tailscale login when it came through Serve, which strips any copy the caller sent, otherwise
        "this Mac". Off HTTPS, its address. A program on this Mac can send Serve's headers itself and
        so name any login: this Mac needs no login to use the page anyway, so the name is a label, not
        proof of who pressed a button."""
        if self.server.https:
            return self._login() if self._through_serve() else THIS_MAC
        return self.client_address[0]

    def _refuse_cross_site(self) -> str | None:
        site = self.server.site
        if self.headers.get_content_type() != "application/json":
            return f"{site.posts} must be sent as application/json."
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc.lower() not in self.server.hosts:
            return f"{site.posts} are accepted only from the {site.page} itself."
        return None

    def _json(self, status: HTTPStatus, body: dict) -> None:
        # allow_nan=False: Python would write NaN and infinities as bare words JSON.parse rejects, and the
        # page would show a dead link. It raises on one instead.
        self._send(status, json.dumps(body, allow_nan=False).encode(), "application/json")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, send_s: float | None = None) -> None:
        """Send one response. One with an error status closes the connection, whatever refused the
        request: its body may not have been read, and must never be parsed as further requests, which
        would skip every check above. `send_s` is how long the whole body may take to send, for one
        too big to send within `timeout`: a socket's timeout bounds a whole send, not each part of it."""
        self._sent = True  # from here, an error can't be answered with another response (`_route`)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.send_header("X-Frame-Options", "DENY")
        if status >= 400:
            self.send_header("Connection", "close")  # sets close_connection, so nothing more is read
        self.end_headers()
        if send_s is None:
            self.wfile.write(body)
            return
        self.connection.settimeout(max(send_s, self.timeout))
        try:
            self.wfile.write(body)
        finally:
            self.connection.settimeout(self.timeout)

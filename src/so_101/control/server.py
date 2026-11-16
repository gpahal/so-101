"""The control page's web server: the standard library's ThreadingHTTPServer, in a daemon thread.

It never touches hardware. Its handlers only put commands on the controller's queue with `submit`
and read `status` and `events`, so the control loop stays the one thread that drives the arm.

    GET  /               the page (page.html beside this file)
    GET  /status         the controller's status as this page sees it, plus the server's: task, bind
                         warning and server_id, a random id per `so101 serve` run, so the page can tell
                         a restarted server from the one it was watching. A poll from the page in
                         control is the link's heartbeat; it sends X-SO101-Visible (link.py)
    GET  /events?after=N the run's events numbered above N, oldest first (at most the newest 50)
    GET  /camera/<name>  the camera's latest preview, one JPEG 320 px wide, from the frames the control
                         loop already read. The server never opens a camera. The page asks for one at
                         the preview rate, so no request stays open to use up the browser's connections.
                         A camera missing from the loop's last handover, or a handover over 2 s old,
                         has no frame (previews.py)
    POST /command        {"command": ..., "outcome"?: ..., "label"?: ..., "for_attempt"?: ...}.
                         Any other key is refused. Waits for the tick that applies it, then replies
                         {"command", "accepted", "reason", "status"}, where "command" is the pause or
                         resume a toggle became. The commands `check` (Run checks: `so101
                         check` on the devices the control loop has open) and `pose` (Check pose: the
                         follower's pose against the demos' start pose) wait longer, and add "summary"
                         and "results", where each result is {"group", "status": PASS, WARN or FAIL,
                         "detail", "fix"}

Each page load makes itself a random client id, held only in memory, and sends it as the
X-SO101-Client header on /status and /command, so the controller knows which page is
in control. A reload, a duplicated tab or another device is a new page, so it views until it takes
control. The id isn't a secret: any page may take control. It only keeps two pages from driving the
robot at once by accident.

Where it listens, who it answers and how it refuses the rest (Host, Origin, framing, and
Tailscale's Serve and Funnel routes, watched every ROUTES_EVERY_S) are `web/server.py`'s, which the
reviewer shares. The caller is named in the status, the page's Viewing banner and the control
events: by its login, "this Mac" for a local request, or its address when the page isn't behind
HTTPS.
"""

import secrets
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeout
from http import HTTPStatus
from importlib import resources
from urllib.parse import SplitResult, parse_qs, unquote

from so_101.control.commands import OPTIONS, TOOLS
from so_101.web import server as web
from so_101.web.server import Bind

SITE = web.Site(page="control page", command="so101 serve", posts="Commands", guards="the arm's controls",
                remote="remote operator", thread="control-page")
COMMAND_WAIT_S = 2.0  # a command waits one tick, 33 ms at 30 fps; longer means the loop is stuck
# Run checks samples the cameras for 2.5 s and reads the arms' motors: about 3 s, with room for a
# slow bus or a camera that takes long to answer.
TOOL_WAIT_S = 30.0
CLIENT_HEADER = "X-SO101-Client"
VISIBLE_HEADER = "X-SO101-Visible"  # "0" when the page polling is out of sight
MAX_CLIENT = 64  # characters: the page sends 32


def resolve_bind(control) -> Bind:
    """Where the control page listens for `control`, so101.toml's `[control]` section: on its `port`,
    by its `bind` (`web.server.resolve_bind`). A TCP forward to the port while it would listen on
    127.0.0.1 raises ValueError."""
    return web.resolve_bind(control.bind, control.port, SITE)


def route_warning(bind: Bind, port: int) -> str | None:
    """What makes a request to the control page unsafe now (`web.server.route_warning`), or None.
    Raises RuntimeError when Tailscale's status can't be read."""
    return web.route_warning(bind, port, SITE)


def page() -> bytes:
    return resources.files("so_101.control").joinpath("page.html").read_bytes()


class ControlServer(web.PageServer):
    def __init__(self, bind: Bind, port: int, *, task: str,
                 routes: Callable[[], str | None]) -> None:
        """Binds now, so a port in use fails before the arm and policy load. Set `controller` before
        `start`. `bind` is where to listen, from `resolve_bind`: the page's status shows its
        warnings. `routes()` is asked before the first request and every ROUTES_EVERY_S on a thread
        of its own, never the control loop's, what makes a request without a login unsafe
        (`route_warning`)."""
        super().__init__(bind, port, Handler, site=SITE, routes=routes)
        self.controller = None
        self.task = task
        self.id = secrets.token_hex(8)  # tells the page a restarted server from the one it watched

    def status(self, client: str | None = None) -> dict:
        return {**self.controller.status(client), "task": self.task, "server_warning": self.warning,
                "server_id": self.id}


class Handler(web.Handler):
    server: ControlServer
    POSTS = ("/command",)

    def get(self, url: SplitResult) -> None:
        if url.path == "/":
            self._send(HTTPStatus.OK, page(), "text/html; charset=utf-8")
        elif url.path == "/status":
            client = self._client()
            if client:  # a page that says it's hidden doesn't keep the link alive
                self.server.controller.heartbeat(client, self.headers.get(VISIBLE_HEADER) != "0")
            self._json(HTTPStatus.OK, self.server.status(client))
        elif url.path == "/events":
            try:
                after = int(parse_qs(url.query).get("after", ["0"])[0])
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "after must be a whole number."})
                return
            self._json(HTTPStatus.OK, {"events": self.server.controller.events(after, 50)})
        elif url.path.startswith("/camera/"):
            self._camera(unquote(url.path.removeprefix("/camera/")))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"No {url.path} here."})

    def _camera(self, name: str) -> None:
        """The camera's latest preview, as one JPEG."""
        previews = self.server.controller.previews
        if previews is None or name not in previews.names:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"No camera preview called {name!r}."})
        elif (jpeg := previews.snapshot(name)) is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": f"No frame from {name} yet."})
        else:
            self._send(HTTPStatus.OK, jpeg, "image/jpeg")

    def post(self, url: SplitResult) -> None:
        try:
            body = self._read_json()
            if not isinstance(body.get("command"), str):
                raise ValueError("no command")
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": 'Send JSON like {"command": "pause"}.'})
            return
        options = {key: value for key, value in body.items() if key != "command"}
        # A key no command takes is refused here, with the connection closed; the loop refuses an
        # option the command it came with doesn't take (`commands.COMMANDS`).
        if unknown := set(options) - set(OPTIONS):
            self._json(HTTPStatus.BAD_REQUEST, {
                "error": f"A command takes no {', '.join(sorted(unknown))}: only {' and '.join(OPTIONS)}."})
            return
        client = self._client()
        future = self.server.controller.submit(body["command"], source="page", client=client, who=self._who(),
                                               **options)
        wait = TOOL_WAIT_S if body["command"] in TOOLS else COMMAND_WAIT_S
        try:
            result = future.result(timeout=wait)
        except FutureTimeout:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "accepted": False, "status": self.server.status(client),
                "reason": f"The control loop didn't answer in {wait:g} s. The command may still apply.",
            })
            return
        reply = {"command": result.command, "accepted": result.accepted, "reason": result.reason}
        if result.report is not None:
            reply |= {"summary": result.report.summary(), "results": result.report.to_json()}
        self._json(HTTPStatus.OK, {**reply, "status": self.server.status(client)})

    def _client(self) -> str | None:
        """The page's own id, from its header, or None without a usable one."""
        client = (self.headers.get(CLIENT_HEADER) or "").strip()
        return client if 0 < len(client) <= MAX_CLIENT else None

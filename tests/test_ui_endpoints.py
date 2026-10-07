"""The UI server's steering endpoints, over a real socket.

The page only ever talks to these, so they get the same treatment the page
would: compose, twist a knob, undo, redo, read the knobs, read the baseline.
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from riddim.direct import knobs
from riddim.params import SongParams
from riddim.ui.server import Handler, State


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.loads(r.read())


def _post(base, path, body):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_steering_endpoints_end_to_end():
    state = State(SongParams(profile="dubstep_140", seed=23))
    Handler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # The page loads and gets a song that can steer.
        song = _get(base, "/api/song")
        assert song["can_undo"] is False and song["can_redo"] is False

        # Every knob the engine knows about - including the key - is served.
        knobs_resp = _get(base, "/api/knobs")
        paths = {k["path"] for k in knobs_resp["knobs"]}
        assert paths == set(knobs(state.params))
        assert "key" in paths
        assert next(k for k in knobs_resp["knobs"] if k["path"] == "key")["kind"] == "key"

        # Twist two knobs; the readback says exactly which moved.
        mutated = _post(base, "/api/mutate",
                        {"ops": [{"path": "tension", "to": 0.8},
                                 {"path": "bass.density", "to": 0.4}]})
        assert mutated["can_undo"] is True
        assert {d["path"] for d in mutated["changed"]["knobs"]} == {"tension", "bass.density"}

        # Word intents go through the same door.
        by_words = _post(base, "/api/mutate", {"ops": ["busier"]})
        assert by_words["changed"]["moved"]

        # Undo walks back both steps; the readback is only empty at the base.
        undone = _post(base, "/api/undo", {})
        assert undone["changed"]["knobs"]  # the first twist is still live
        undone = _post(base, "/api/undo", {})
        assert undone["changed"]["knobs"] == [] and undone["can_redo"] is True

        # Redo walks forward again.
        redone = _post(base, "/api/redo", {})
        assert redone["changed"]["knobs"]

        # The baseline is the start of the take, with its own knob values.
        baseline = _get(base, "/api/baseline")
        assert baseline["knobs"]
        assert "key" in {k["path"] for k in baseline["knobs"]}
        assert baseline["seed"] == song["seed"]
    finally:
        server.shutdown()
        server.server_close()
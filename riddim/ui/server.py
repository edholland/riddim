"""A small local web app for previewing and steering the engine.

Standard library only - `http.server` and JSON - so nothing new to install.
The page is a single HTML file; the server holds one `SongParams` in memory
and every interaction is "edit params, recompose". Because params resolve
completely and each part has its own seed, the page can change one slider,
or regenerate one part, and show exactly what moved.

    riddim-cli ui            # then open http://localhost:8765
"""

from __future__ import annotations

import io
import json
import mimetypes
import random
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import profile as profile_mod
from ..compose import compose
from ..params import SongParams
from ..song import Rig, Song
from ..theory import harmony
from ..theory.groove import FEELS
from ..units import TICKS_PER_BAR

HERE = Path(__file__).parent


class State:
    """The one song the page is editing."""

    def __init__(self, params: SongParams | None = None):
        self.lock = threading.Lock()
        self.params = params or SongParams(profile="dubstep_140", seed=random.randrange(1, 10**6))
        self.song: Song = compose(params=self.params)

    def recompose(self, params: SongParams) -> Song:
        with self.lock:
            self.params = params
            self.song = compose(params=self.params)
            return self.song

    def regenerate(self, part: str) -> Song:
        from ..compose import regenerate
        with self.lock:
            self.song = regenerate(self.song, part)
            self.params = self.song.params
            return self.song

    def reroll(self, keep: dict) -> Song:
        """A new song: fresh seed, everything drawn again except what the
        page asked to keep (profile, key, bpm...)."""
        with self.lock:
            fresh = SongParams(profile=self.params.profile, seed=random.randrange(1, 10**6))
            for k, v in keep.items():
                if hasattr(fresh, k) and v not in (None, ""):
                    setattr(fresh, k, v)
            self.params = fresh
            self.song = compose(params=self.params)
            return self.song


def song_to_dict(song: Song) -> dict:
    """Everything the page needs to draw and play the song."""
    arr = song.arrangement
    return {
        "name": song.name,
        "bpm": song.bpm,
        "bars": song.bars,
        "seed": song.seed,
        "key": str(song.scale),
        "progression": song.harmony.progression.name,
        "chords": [
            {"symbol": ev.chord.symbol(), "start": ev.start, "length": ev.length,
             "pitches": list(ev.chord.pitches)}
            for ev in song.harmony.events
        ],
        "concept": song.concept,
        "notes_for_player": song.notes_for_player,
        "sections": [
            {"name": s.name, "bars": s.bars, "start_bar": s.start_bar,
             "energy": s.energy, "active": list(s.active), "dub_throw": s.dub_throw,
             "direction": s.direction}
            for s in arr.sections
        ],
        "parts": {
            p.role: {
                "output": p.output, "channel": p.channel,
                "notes": [[n.midi, n.start, n.length, n.velocity] for n in p.sorted_notes()],
                "range": list(p.range()),
            }
            for p in song.parts
        },
        "params": song.params.to_dict(),
        "ticks_per_bar": TICKS_PER_BAR,
    }


def options() -> dict:
    return {
        "profiles": list(profile_mod.PROFILES),
        "feels": list(FEELS),
        "lead_styles": ["call_response", "long_tones", "riff"],
        "contours": ["static", "arch", "valley", "ascending", "descending",
                     "zigzag", "terraced", "falling_tail"],
        "progressions": sorted(harmony.BY_NAME),
        "roles": ["drums", "perc", "sub", "chords", "pad", "lead"],
        "modes": ["minor", "dorian", "phrygian", "harmonic_minor", "phrygian_dominant"],
        "roots": ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
    }


class Handler(BaseHTTPRequestHandler):
    state: State  # set on the class by serve()

    # -- plumbing ------------------------------------------------------

    def log_message(self, fmt, *args):  # quieter
        if "/api/" in (args[0] if args else ""):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _file(self, path: Path, content_type: str | None = None):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- routes --------------------------------------------------------

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                return self._file(HERE / "index.html", "text/html; charset=utf-8")
            if self.path == "/api/song":
                return self._json(song_to_dict(self.state.song))
            if self.path == "/api/options":
                return self._json(options())
            if self.path == "/api/progressions":
                scale = self.state.song.scale
                items = [
                    {"name": pr.name, "tension": round(pr.tension, 2),
                     "chords": " ".join(
                         harmony.realise(scale, pr, len(pr.degrees), 1).symbols())}
                    for pr in harmony.enumerate_progressions(scale)
                ]
                return self._json({"key": str(scale), "progressions": items})
            if self.path == "/api/ports":
                import mido
                return self._json({"outputs": mido.get_output_names()})
            if self.path == "/api/midi":
                from ..render import midifile
                import tempfile, os
                fd, tmp = tempfile.mkstemp(suffix=".mid")
                os.close(fd)
                midifile.write(self.state.song, tmp)
                data = Path(tmp).read_bytes()
                os.unlink(tmp)
                self.send_response(200)
                self.send_header("Content-Type", "audio/midi")
                self.send_header("Content-Disposition", f'attachment; filename="{self.state.song.name}.mid"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)
        except Exception as exc:  # surface to the page rather than a blank 500
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/compose":
                params = SongParams.from_dict(body["params"])
                return self._json(song_to_dict(self.state.recompose(params)))
            if self.path == "/api/regenerate":
                return self._json(song_to_dict(self.state.regenerate(body["part"])))
            if self.path == "/api/reroll":
                return self._json(song_to_dict(self.state.reroll(body.get("keep", {}))))
            if self.path == "/api/send":
                return self._json(self._send(body))
            self.send_error(404)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def _send(self, body: dict) -> dict:
        from ..deluge import render
        from ..deluge.client import Deluge

        rows = body.get("template_rows") or None
        template = render.parse_template(rows) if rows else None
        with Deluge(output=body.get("port"), input=body.get("port")) as deluge:
            if not deluge.ping():
                return {"error": "Deluge did not answer the ping"}
            report = render.send(
                self.state.song, deluge, mode=body.get("mode", "jam"),
                template_rows=template, progress=False,
            )
        return {"clips": report.clips, "notes": report.notes, "trimmed": report.trimmed,
                "failures": report.failures}


def serve(port: int = 8765, open_browser: bool = True, params: SongParams | None = None):
    Handler.state = State(params)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"riddim ui at {url}  (ctrl-c to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

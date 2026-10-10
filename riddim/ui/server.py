"""A small local web app for previewing and steering the engine.

Standard library only - `http.server` and JSON - so nothing new to install.
The page is a single HTML file; the server holds one `SongParams` in memory
and every interaction is "edit params, recompose". Because params resolve
completely and each part has its own seed, the page can change one slider,
or regenerate one part, and show exactly what moved.

    riddim-cli ui            # then open http://127.0.0.1:8765

The steering model (see riddim.direct):

* every state change is a *step* on a history trail, so the page can
  undo / redo and diff against the start of the take;
* twisting a knob keeps every seed frozen - the result is a small move
  from where you were, not a new roll;
* "re-roll" (a part or the whole song) starts a *new take*, which becomes
  the A/B baseline.
"""

from __future__ import annotations

from dataclasses import asdict

import io
import json
import mimetypes
import random
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import critique, profile as profile_mod
from ..compose import compose
from ..direct import Op, apply_ops, diff as params_diff, direct as parse_direct
from ..direct import knobs as read_knobs
from ..direct import song_diff
from ..params import SongParams
from ..song import Rig, Song
from ..theory import harmony
from ..theory.groove import FEELS
from ..units import TICKS_PER_BAR

HERE = Path(__file__).parent

HISTORY_LIMIT = 60  # a take is a few dozen twists at most


class _Snap:
    """One point on the steering trail: the params and song that were live."""

    __slots__ = ("params", "song")

    def __init__(self, params: SongParams, song: Song):
        self.params = params
        self.song = song


def _ops(body: dict) -> list[Op]:
    """Turn the page's requested changes into ops.

    The page sends absolute targets from its sliders: {"path", "to"} or
    relative nudges: {"path", "delta"}.
    """
    ops: list[Op] = []
    for item in body.get("ops") or []:
        if isinstance(item, str):
            ops.extend(parse_direct(item))
        elif isinstance(item, dict) and item.get("path"):
            ops.append(Op(
                item["path"],
                delta=float(item.get("delta", 0.0)),
                to=item.get("to"),
            ))
    return ops


class State:
    """The one song the page is steering, with its trail."""

    def __init__(self, params: SongParams | None = None):
        self.lock = threading.Lock()
        self.params = params or SongParams(profile="dubstep_140", seed=random.randrange(1, 10**6))
        self.song: Song = compose(params=self.params)
        # The take: the song twists are steering, and the params it started from.
        self.baseline: Song = self.song
        self.base_params: SongParams = self.params
        self.history: list[_Snap] = []
        self.future: list[_Snap] = []

    # -- take bookkeeping ------------------------------------------------

    def _push(self) -> None:
        self.history.append(_Snap(self.params, self.song))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)
        self.future.clear()

    def _retake(self, params: SongParams, song: Song) -> Song:
        self._push()
        self.params, self.song = params, song
        self.baseline, self.base_params = song, params
        return song

    # -- the steering ------------------------------------------------------

    def recompose(self, params: SongParams) -> Song:
        """A wholesale params edit (arrange/advanced panel) - a step, not a retake."""
        with self.lock:
            self._push()
            self.params = params
            self.song = compose(params=params)
            return self.song

    def mutate(self, ops: list[Op]) -> Song:
        """Twist knobs: apply ops, keep every seed frozen, recompose."""
        with self.lock:
            if not ops:
                return self.song
            self._push()
            self.params = apply_ops(self.params, ops)
            self.song = compose(params=self.params)
            return self.song

    def regenerate(self, part: str) -> Song:
        """A fresh roll of one part - that starts a new take."""
        from ..compose import regenerate
        with self.lock:
            song = regenerate(self.song, part)
            return self._retake(song.params, song)

    def reroll(self, keep: dict) -> Song:
        """A new song: fresh seed, everything drawn again except what the
        page asked to keep (profile, key, bpm...). Starts a new take."""
        with self.lock:
            fresh = SongParams(profile=self.params.profile, seed=random.randrange(1, 10**6))
            for k, v in keep.items():
                if hasattr(fresh, k) and v not in (None, ""):
                    setattr(fresh, k, v)
            return self._retake(fresh, compose(params=fresh))

    def undo(self) -> Song:
        with self.lock:
            if not self.history:
                return self.song
            self.future.append(_Snap(self.params, self.song))
            snap = self.history.pop()
            self.params, self.song = snap.params, snap.song
            return self.song

    def redo(self) -> Song:
        with self.lock:
            if not self.future:
                return self.song
            self.history.append(_Snap(self.params, self.song))
            snap = self.future.pop()
            self.params, self.song = snap.params, snap.song
            return self.song

    def revise(self, max_steps: int = 8) -> Song:
        """The critic's revision loop, run as the page's take model.

        Twist fixes go through mutate (steps on the trail - undoable), roll
        fixes through regenerate (a new take). A step that does not improve
        the score is undone for twists - the take must not rot.
        """
        for _ in range(max_steps):
            chosen = critique.next_fix(self.song)
            if chosen is None:
                return self.song
            _finding, fix = chosen
            before = critique.score(self.song)
            if fix.kind == "twist":
                # Try it on a copy first: a twist that does not improve the
                # take is never applied, so the trail holds no dead steps.
                trial = compose(params=apply_ops(self.params, list(fix.ops)))
                if critique.score(trial) - before < 0.01:
                    return self.song
                self.mutate(list(fix.ops))
            else:
                # A roll is a new take either way - the A/B baseline moves,
                # and the maker can compare or reset by hand.
                self.regenerate(fix.part)
                if critique.score(self.song) - before < 0.01:
                    return self.song
        return self.song


def _changed(state: State, song: Song) -> dict:
    """What moved since the take began - for the page's readback.

    `note_changes` counts notes added and removed per part, which is the
    note-level picture the piano-roll shading needs: a key change moves every
    note, a density nudge adds a handful of hats.
    """
    base = state.baseline
    if base is song:
        counts = _part_counts(song)
        return {"knobs": [], "moved": [],
                "notes": {r: {"base": n, "now": n} for r, n in counts.items()},
                "note_changes": {}}
    deltas = [
        {"path": d.path, "from": round(d.from_value, 3), "to": round(d.to_value, 3)}
        for d in params_diff(state.base_params, state.params)
    ]
    moved = song_diff(base, song).moved_parts
    before = _part_counts(base)
    after = _part_counts(song)
    return {
        "knobs": deltas,
        "moved": moved,
        "notes": {r: {"base": before.get(r, 0), "now": after.get(r, 0)}
                  for r in set(before) | set(after)},
        "note_changes": _note_changes(base, song),
    }


def _note_changes(base: Song, song: Song) -> dict:
    def _sets(part) -> set:
        return {(n.midi, n.start, n.length, n.velocity) for n in part.notes}

    b = {p.role: _sets(p) for p in base.parts}
    t = {p.role: _sets(p) for p in song.parts}
    out = {}
    for role in set(b) | set(t):
        added = len(t.get(role, set()) - b.get(role, set()))
        removed = len(b.get(role, set()) - t.get(role, set()))
        if added or removed:
            out[role] = {"added": added, "removed": removed}
    return out


def _part_counts(song: Song) -> dict:
    return {p.role: len(p.notes) for p in song.parts}


def song_to_dict(song: Song, state: State | None = None) -> dict:
    """Everything the page needs to draw and play the song."""
    arr = song.arrangement
    d = {
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
             "direction": s.direction,
             "progression": s.progression or song.harmony.progression.name,
             "ramp": s.ramp}
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
        "checks": [asdict(f) for f in critique.critique(song)],
        "score": round(critique.score(song), 2),
    }
    if state is not None:
        d["can_undo"] = bool(state.history)
        d["can_redo"] = bool(state.future)
        d["changed"] = _changed(state, song)
    return d


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
        if args and "/api/" in str(args[0]):
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
                return self._json(song_to_dict(self.state.song, self.state))
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
            if self.path == "/api/knobs":
                knobs = [asdict(k) for k in read_knobs(self.state.params).values()]
                return self._json({"knobs": knobs, "key": str(self.state.song.scale)})
            if self.path == "/api/baseline":
                d = song_to_dict(self.state.baseline)
                d["knobs"] = [asdict(k) for k in read_knobs(self.state.base_params).values()]
                return self._json(d)
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
                return self._json(song_to_dict(self.state.recompose(params), self.state))
            if self.path == "/api/mutate":
                song = self.state.mutate(_ops(body))
                return self._json(song_to_dict(song, self.state))
            if self.path == "/api/regenerate":
                return self._json(song_to_dict(self.state.regenerate(body["part"]), self.state))
            if self.path == "/api/reroll":
                return self._json(song_to_dict(self.state.reroll(body.get("keep", {})), self.state))
            if self.path == "/api/undo":
                return self._json(song_to_dict(self.state.undo(), self.state))
            if self.path == "/api/redo":
                return self._json(song_to_dict(self.state.redo(), self.state))
            if self.path == "/api/revise":
                steps = int(body.get("max_steps", 8))
                return self._json(song_to_dict(self.state.revise(steps), self.state))
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

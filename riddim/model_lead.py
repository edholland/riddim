"""Leads written by a trained model over riddim's band.

riddim keeps everything it is good at - the structure, the drums, the
sub, the chords, the live-set and Deluge plumbing - and hands the hook to
the Anticipatory Music Transformer (fine-tuned on the open MIDI that plays
most like early dubstep and dub). The model hears the band as it will be
played and writes the lead in the drops:

* the model drafts several four-bar phrases over the first drop's band
  (having heard the 8 bars that lead into it) and riddim keeps the one
  that works best as a hook: on the chords where it lands on the beat, a
  tight register, small leaps, a rhythm that repeats inside itself;
* riddim arranges the drop around it - hook, hook, an answer, hook - and
  the model writes the answer having heard the hook twice;
* every later drop brings the same hook back, bent onto its own chords
  where they differ, with fresh answers - developed, not re-rolled;
* the lead stays in the register riddim's lead used there, in the song's
  mode, at most one onset per eighth, sampled on the sixteenth grid.

The model runs in its own Python (see ml/): riddim talks to a worker over
JSON lines, so this module needs nothing beyond the standard library.

    RIDDIM_ML_PYTHON   the ML environment's python  (~/ml/venv/bin/python)
    RIDDIM_ML_MODEL    the model directory          (v2/best, else the first fine-tune, else base)
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

from .song import Note, Song
from .theory import harmony as harmony_mod
from .units import TICKS_PER_16TH, TICKS_PER_BAR, TICKS_PER_BEAT

ML_HOME = Path(os.environ.get("ML_HOME", Path.home() / "ml"))
WORKER = Path(__file__).resolve().parent.parent / "ml" / "lead_worker.py"

LEAD_INSTR = 81                               # GM sawtooth lead
PROGRAMS = {"sub": 38, "chords": 4, "pad": 89, "perc": 128, "drums": 128}
CONTEXT_BARS = 8                              # what the model hears before it writes
MAX_SECONDS = 95.0                            # the model's 100 s horizon, with room
DROP_ENERGY = 0.8
PHRASE_BARS = 4
HOOK_CANDIDATES = 6
ANSWER_CANDIDATES = 3
FORM = "AABA"                                 # per four phrases of a drop


def default_python() -> Path:
    return Path(os.environ.get("RIDDIM_ML_PYTHON", ML_HOME / "venv" / "bin" / "python"))


def default_model() -> Path:
    if os.environ.get("RIDDIM_ML_MODEL"):
        return Path(os.environ["RIDDIM_ML_MODEL"])
    for name in ("v2/best", "ft_dmz/final", "ft_dmz/latest", "music-medium-800k"):
        path = ML_HOME / "models" / name
        if (path / "config.json").exists():
            return path
    return ML_HOME / "models" / "music-medium-800k"


def unavailable() -> str | None:
    """Why the model can't run here, or None if it can."""
    if not default_python().exists():
        return f"no ML python at {default_python()} (set RIDDIM_ML_PYTHON)"
    if not (default_model() / "config.json").exists():
        return f"no model at {default_model()} (set RIDDIM_ML_MODEL)"
    if not WORKER.exists():
        return f"missing {WORKER}"
    return None


class ModelError(RuntimeError):
    pass


class Writer:
    """A running worker. Loads the model once; requests are serialised."""

    def __init__(self, python: Path | None = None, model: Path | None = None):
        self.python = python or default_python()
        self.model = model or default_model()
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.cache: dict[str, list] = {}

    def start(self) -> "Writer":
        if self.proc is not None and self.proc.poll() is None:
            return self
        why = unavailable() if self.python == default_python() else None
        if why:
            raise ModelError(why)
        self.proc = subprocess.Popen(
            [str(self.python), str(WORKER), str(self.model)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        hello = self._read()
        if not hello.get("ready"):
            raise ModelError(f"worker did not start: {hello}")
        return self

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None

    def __enter__(self) -> "Writer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise ModelError("the model worker exited")
        return json.loads(line)

    def request(self, req: dict) -> dict:
        key = json.dumps(req, sort_keys=True)
        if key in self.cache:
            return {"notes": self.cache[key], "cached": True}
        with self.lock:
            self.start()
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            reply = self._read()
        if "error" in reply:
            raise ModelError(reply["error"])
        self.cache[key] = reply["notes"]
        return reply


_shared: Writer | None = None


def shared() -> Writer:
    """One worker per process (the UI keeps it warm between presses)."""
    global _shared
    if _shared is None:
        _shared = Writer()
        atexit.register(_shared.close)
    return _shared


# -- song <-> the model's events ---------------------------------------------

def _instr(part) -> int:
    if part.output == "kit":
        return 128
    return PROGRAMS.get(part.role, 0)


def _events(song: Song, t0: int, t1: int, shift: int, spt: float,
            lead_from: int | None = None) -> list:
    """Notes in [t0, t1) moved by `shift` ticks, as [instr, pitch, start_s, dur_s].
    Lead notes only from `lead_from` on (the model's own), none if None."""
    from .render.midifile import GM_DRUMS

    out = []
    for part in song.parts:
        is_lead = part.role == "lead"
        if is_lead and lead_from is None:
            continue
        instr = LEAD_INSTR if is_lead else _instr(part)
        for n in part.notes:
            if t0 <= n.start < t1 and not (is_lead and n.start < lead_from):
                pitch = GM_DRUMS.get(n.midi, n.midi) if instr == 128 else n.midi
                out.append([instr, pitch, round((n.start + shift) * spt, 4),
                            round(min(n.length, t1 - n.start) * spt, 4)])
    return out


def _to_grid(notes, origin: int, spt: float, end: int, velocity: int) -> list[Note]:
    """The model's seconds -> riddim notes: on the sixteenth grid,
    monophonic, the downbeats leaned on."""
    placed: dict[int, tuple[int, int]] = {}
    for s, d, p in notes:
        tick = origin + int(round(s / spt / TICKS_PER_16TH)) * TICKS_PER_16TH
        if origin <= tick < end:
            placed.setdefault(tick, (int(p), max(1, int(round(d / spt / TICKS_PER_16TH)))))
    ticks = sorted(placed)
    out = []
    for i, tick in enumerate(ticks):
        p, steps = placed[tick]
        nxt = ticks[i + 1] if i + 1 < len(ticks) else end
        dur = min(steps * TICKS_PER_16TH, nxt - tick, end - tick)
        strong = tick % TICKS_PER_BEAT == 0
        vel = min(127, velocity + 10) if strong else velocity - 4
        out.append(Note(p, tick, max(TICKS_PER_16TH // 2, dur), vel))
    return out


# -- judging a phrase -----------------------------------------------------------

def _onsets(notes: list[Note], t0: int) -> list[set]:
    bars: dict[int, set] = {}
    for n in notes:
        bars.setdefault((n.start - t0) // TICKS_PER_BAR, set()).add((n.start - t0) % TICKS_PER_BAR)
    return [bars.get(i, set()) for i in range(max(bars) + 1)] if bars else []


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def phrase_score(notes: list[Note], song: Song, t0: int, bars: int) -> float:
    """How well a few bars work as a hook. Higher is better; 0 is unusable."""
    if len(notes) < 2:
        return 0.0
    per_bar = len(notes) / bars
    fit = [harmony_mod.sits_on(n.midi, ev.chord) for n in notes
           if n.start % TICKS_PER_BEAT == 0 and (ev := song.harmony.at(n.start)) is not None]
    all_fit = [harmony_mod.sits_on(n.midi, ev.chord) for n in notes
               if (ev := song.harmony.at(n.start)) is not None]
    span = max(n.midi for n in notes) - min(n.midi for n in notes)
    leaps = [abs(b.midi - a.midi) for a, b in zip(notes, notes[1:])]
    mean_leap = sum(leaps) / len(leaps) if leaps else 0
    bar_onsets = _onsets(notes, t0) + [set()] * bars
    bar_onsets = bar_onsets[:bars]
    # a hook repeats its own rhythm: bar 1 against 3, 2 against 4 (or 1 against 2)
    half = bars // 2 or 1
    rhythm = [_jaccard(bar_onsets[i], bar_onsets[i + half]) for i in range(bars - half)]
    empty = sum(1 for b in bar_onsets if not b)
    repeats = sum(1 for x in leaps if x == 0) / len(leaps) if leaps else 1.0
    distinct = len({n.midi for n in notes})
    score = 3.0
    score += 2.0 * (sum(fit) / len(fit) if fit else 0.5)
    score += 1.0 * sum(all_fit) / len(all_fit) if all_fit else 0.0
    score += 1.5 * (sum(rhythm) / len(rhythm) if rhythm else 0.0)
    score -= 0.6 * max(0.0, abs(per_bar - 3.5) - 1.5)          # 2-5 a bar is the zone
    score -= 0.25 * max(0, span - 12)                           # an octave, give or take
    score -= 0.4 * max(0.0, mean_leap - 4.0)
    score -= 0.8 * max(0, empty - 1)
    score -= 3.0 * max(0.0, repeats - 0.4)                      # a tune, not a pedal
    score -= 1.0 * max(0, 3 - distinct)
    last = notes[-1]
    ev = song.harmony.at(last.start)
    if ev is not None and last.midi % 12 in ev.chord.pitch_classes():
        score += 0.4                                            # it lands
    return max(0.01, score)


def bend(notes: list[Note], song: Song, lo: int, hi: int) -> list[Note]:
    """A restated hook over the chords where it now sits: a note on the beat
    that no longer sits on its chord moves to the nearest chord tone."""
    scale = set(song.scale.pitch_classes)
    out = []
    for n in notes:
        ev = song.harmony.at(n.start)
        if ev is None or n.start % TICKS_PER_BEAT or harmony_mod.sits_on(n.midi, ev.chord):
            out.append(n)
            continue
        pcs = ev.chord.pitch_classes() & scale or ev.chord.pitch_classes()
        options = [m for m in range(max(0, lo - 2), min(127, hi + 2) + 1) if m % 12 in pcs]
        best = min(options, key=lambda m: (abs(m - n.midi), m)) if options else n.midi
        out.append(replace(n, midi=best))
    return out


def drops(song: Song) -> list:
    return [s for s in song.arrangement.sections if s.energy >= DROP_ENERGY and s.plays("lead")]


def _form(phrases: int) -> str:
    if phrases <= 1:
        return "A"
    if phrases == 2:
        return "AB"
    if phrases == 3:
        return "AAB"
    return (FORM * (phrases // 4 + 1))[:phrases]


class _Job:
    """One song's worth of model calls."""

    def __init__(self, song: Song, writer: Writer, take: int, top_p: float, progress,
                 style: str = "rhythm"):
        self.song, self.writer, self.take, self.top_p, self.progress = song, writer, take, top_p, progress
        self.style = style
        self.spt = 60.0 / song.bpm / TICKS_PER_BEAT
        self.scale = sorted(set(song.scale.pitch_classes))
        self.seed = (song.params.seed if song.params is not None else song.seed) * 1000 + take * 7919
        self.calls = 0
        lead = song.part("lead")
        self.vel = round(sum(n.velocity for n in lead.notes) / len(lead.notes)) if lead.notes else 90
        self.work = replace(song, parts=list(song.parts))
        self.lead = replace(lead, notes=list(lead.notes))
        self.work.parts[self.work.parts.index(lead)] = self.lead

    def draft(self, a: int, b: int, lo: int, hi: int, lead_from: int | None, seed: int) -> list[Note]:
        """The model's lead for [a, b), having heard up to 8 bars before."""
        c0 = max(0, a - CONTEXT_BARS * TICKS_PER_BAR)
        ctx = a - c0
        spt = self.spt
        events = _events(self.work, c0, a, -c0, spt, lead_from)
        events += _events(self.work, a, b, ctx - a, spt, None)
        req = {
            "op": "lead", "events": events,
            "start": round(ctx * spt, 4), "end": round((ctx + b - a) * spt, 4),
            "lo": lo, "hi": hi, "scale": self.scale, "max_repeat": 3,
            "top_p": self.top_p, "seed": seed,
        }
        if self.style == "rhythm":
            # riddim's phrasing, the model's pitches
            mine = sorted((n for n in self.song.part("lead").notes if a <= n.start < b),
                          key=lambda n: n.start)
            mono = [n for i, n in enumerate(mine) if i == 0 or n.start != mine[i - 1].start]
            if not mono:
                return []
            req["rhythm"] = [[round((n.start - a + ctx) * spt, 4), round(n.length * spt, 4)]
                             for n in mono]
        else:
            req["min_gap"] = round(TICKS_PER_16TH * 2 * spt, 4)
        reply = self.writer.request(req)
        self.calls += 1
        if self.style == "rhythm":
            pitches = [p for _s, _d, p in sorted(reply["notes"])]
            return [replace(n, midi=int(p)) for n, p in zip(mono, pitches)]
        notes = [(s - ctx * spt, d, p) for s, d, p in reply["notes"]]
        return _to_grid(notes, a, spt, b, self.vel)

    def best(self, a, b, lo, hi, lead_from, seeds, bonus=None) -> list[Note]:
        bars = (b - a) // TICKS_PER_BAR
        scored = []
        for seed in seeds:
            notes = self.draft(a, b, lo, hi, lead_from, seed)
            sc = phrase_score(notes, self.work, a, bars) + (bonus(notes) if bonus else 0.0)
            scored.append((sc, notes))
        return max(scored, key=lambda x: x[0])[1]

    def place(self, notes: list[Note]) -> None:
        self.lead.notes.extend(notes)

    def say(self, msg: str) -> None:
        if self.progress:
            self.progress(msg)


STYLES = ("rhythm", "free")


def write(song: Song, writer: Writer | None = None, take: int = 0, top_p: float = 0.9,
          progress=None, style: str = "rhythm") -> Song:
    """A copy of `song` with the model's lead in every drop: a hook the
    model drafted and riddim chose, arranged hook-hook-answer-hook.

    style "rhythm": the model chooses the pitches on riddim's own lead
    rhythm. "free": the model writes the rhythm too (sampled freely, then
    snapped to the sixteenths)."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    writer = writer or shared()
    lead = song.part("lead")
    targets = drops(song)
    if lead is None or not targets:
        return song
    job = _Job(song, writer, take, top_p, progress, style)
    lo_all, hi_all = lead.range() if lead.notes else (64, 88)
    hook: list[Note] | None = None        # relative ticks, from the first drop
    phrase = PHRASE_BARS * TICKS_PER_BAR
    for i, sec in enumerate(targets):
        s0, s1 = sec.start_tick, sec.start_tick + sec.length_ticks
        own = sorted(n.midi for n in lead.notes if s0 <= n.start < s1) or [lo_all, hi_all]
        # around where riddim's lead sits, not its extremes - left the whole
        # span, the model sinks to the bottom of it and the lead turns to mud
        mid = own[len(own) // 2]
        lo, hi = max(own[0], mid - 5), min(own[-1], mid + 9)
        job.lead.notes = [n for n in job.lead.notes if not s0 <= n.start < s1]
        plen = min(phrase, sec.length_ticks)
        n_phrases = max(1, sec.length_ticks // plen)
        form = _form(n_phrases)
        job.say(f"model lead: {sec.name} ({sec.bars} bars) as {form}")
        for k, letter in enumerate(form):
            a = s0 + k * plen
            b = s1 if k == n_phrases - 1 else a + plen
            if letter == "A" and hook is None:
                seeds = [job.seed + i * 101 + c for c in range(HOOK_CANDIDATES)]
                notes = job.best(a, b, lo, hi, None, seeds)
                hook = [n.moved(-a) for n in notes]
                job.say(f"  hook chosen from {HOOK_CANDIDATES} drafts: {len(notes)} notes")
            elif letter == "A":
                notes = bend([n.moved(a) for n in hook if n.start < b - a], job.work, lo, hi)
            else:
                hook_rhythm = _onsets([n.moved(a) for n in hook], a)

                def related(ns, a=a):
                    mine = _onsets(ns, a)
                    pairs = list(zip(mine, hook_rhythm))
                    sim = sum(_jaccard(x, y) for x, y in pairs) / len(pairs) if pairs else 0.0
                    same = [(n.start - a, n.midi) for n in ns] == [(n.start, n.midi) for n in hook]
                    # an answer echoes the hook's rhythm, not its notes
                    return 0.8 * sim - (2.0 if same else 0.0)

                seeds = [job.seed + i * 101 + 50 + k * 7 + c for c in range(ANSWER_CANDIDATES)]
                notes = job.best(a, b, lo, hi, s0, seeds, related)
            job.place(notes)
    job.lead.notes.sort(key=lambda n: (n.start, n.midi))
    job.work.notes_for_player = list(song.notes_for_player) + [
        f"Lead in the drops: a hook drafted by the model ({Path(writer.model).name}, "
        f"take {take}) and arranged by riddim - {job.calls} model calls."]
    return job.work

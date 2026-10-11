"""A show's worth of music: compose many, keep the best, order them as a set.

For each slot in the set, several candidate songs are composed from
consecutive seeds, scored by the listener (`analyze`) and the critic
(`critique`), the best is revised by the critic's own loop, and the winner
is written out three ways:

    NN_title.mid    a standard MIDI file - drop it into a DAW, or audition it
    NN_title.json   its complete params: `riddim-cli send --params` puts the
                    exact same song on the Deluge, and the UI can load it
    NN_title.txt    the plan: key, tempo, structure, hook, player notes

Then the set is *ordered* - a live set is a shape, not a playlist: it opens
in the slow roots tempo, climbs through steppers and half-time into the
140s, and comes home. Within a tempo block, songs are chained by key
signature so each change is at most a step round the circle of fifths
(the Camelot code is printed for anyone mixing by ear), and by energy so the
block climbs. `setlist.md` and `setlist.csv` say all of that.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass, field

from . import analyze, critique
from .compose import compose
from .params import SongParams
from .song import Rig, Song

# The shape of the night: which feels, in which order.
# Sound system to garage to dubstep: the music's own family tree, in order.
SET_ORDER = ("roots_dub", "steppers_dub", "dubby_halftime", "uk_garage",
             "dark_garage", "dubstep_140")

_ADJ = (
    "Iron", "Midnight", "Concrete", "Lion", "Heavy", "Smoke", "Echo", "Dread",
    "Deep", "Black", "Sound", "Steel", "Rebel", "Night", "Northern", "Ghost",
    "Hollow", "Distant", "Burning", "Salt", "Low", "Silent", "Copper", "Signal",
    "Babylon", "Zion", "Tower", "River", "Shadow", "Bassline", "Version", "Dub",
)
_NOUN = (
    "Riddim", "Station", "Prophecy", "Chamber", "Frequency", "Pressure",
    "Warning", "Meditation", "Foundation", "Ritual", "Transmission", "Weight",
    "Tide", "Horizon", "Corridor", "Siren", "Garden", "Engine", "Sermon",
    "Skank", "System", "Crossing", "Movement", "Exodus", "Dispatch", "Furnace",
    "Signal", "Archive", "Lantern", "Monsoon", "Satellite", "Rockers",
)


def title_for(song: Song) -> str:
    """A deterministic name, so a seed always comes back as the same song."""
    rng = random.Random(f"title:{song.params.profile}:{song.seed}")
    adj, noun = rng.choice(_ADJ), rng.choice(_NOUN)
    while noun == adj:
        noun = rng.choice(_NOUN)
    return f"{adj} {noun}"


def camelot(song: Song) -> str:
    """The key's Camelot code by key signature: A minor (no accidentals) is
    8A, each sharp one step clockwise, each flat one step back. Modes are
    placed by their signature - E phrygian shares A minor's notes, so it
    mixes like 8A, which is what an ear doing harmonic mixing hears."""
    n = ((7 + song.scale.sharps) % 12) + 1
    return f"{n}A"


def _signature_distance(a: Song, b: Song) -> int:
    d = abs(a.scale.sharps - b.scale.sharps) % 12
    return min(d, 12 - d)


@dataclass
class Entry:
    song: Song
    title: str
    score: float
    listener: float
    critic: float
    candidates: list = field(default_factory=list)   # (seed, combined score)
    files: dict = field(default_factory=dict)

    @property
    def minutes(self) -> float:
        return self.song.arrangement.duration_seconds(self.song.bpm) / 60.0


def judge(song: Song) -> tuple[float, float, float]:
    """(combined, listener, critic): the listener's ear, checked by the critic."""
    listener = analyze.score(song)
    crit = critique.score(song)
    return round(0.7 * listener + 0.3 * crit, 4), listener, crit


def best_of(profile: str, seeds, rig: Rig | None = None, fix: bool = True,
            key: str | None = None) -> Entry:
    """Compose a song per seed, keep the one that judges best, revise it."""
    tried = []
    best: tuple | None = None
    for seed in seeds:
        song = compose(params=SongParams(profile=profile, seed=seed, key=key), rig=rig)
        combined, listener, crit = judge(song)
        tried.append((seed, combined))
        if best is None or combined > best[0]:
            best = (combined, song)
    song = best[1]
    if fix:
        song, _ = critique.revise(song, rig=rig)
    combined, listener, crit = judge(song)
    return Entry(song, title_for(song), combined, listener, crit, tried)


def order(entries: list[Entry]) -> list[Entry]:
    """Shape a set: tempo blocks in SET_ORDER, and inside each block a chain
    that starts on the calmest song and keeps every key change small."""
    out: list[Entry] = []
    blocks: dict[str, list[Entry]] = {}
    for e in entries:
        blocks.setdefault(e.song.params.profile, []).append(e)
    names = [p for p in SET_ORDER if p in blocks] + [p for p in blocks if p not in SET_ORDER]
    for name in names:
        pool = sorted(blocks[name], key=lambda e: _peak(e.song))
        chain = [pool.pop(0)]
        while pool:
            last = chain[-1].song
            nxt = min(pool, key=lambda e: (_signature_distance(last, e.song),
                                           -e.score))
            pool.remove(nxt)
            chain.append(nxt)
        out += chain
    return out


def _peak(song: Song) -> float:
    return max(s.energy for s in song.arrangement.sections)


def _slug(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def write(entries: list[Entry], out_dir: str, audio: bool = False, progress=None) -> str:
    """Write every entry and the set list. Returns the set list path."""
    from .render import midifile

    os.makedirs(out_dir, exist_ok=True)
    for i, e in enumerate(entries, 1):
        base = os.path.join(out_dir, f"{i:02d}_{_slug(e.title)}")
        e.song.name = e.title
        e.files = {"mid": base + ".mid", "json": base + ".json", "txt": base + ".txt"}
        midifile.write(e.song, base + ".mid")
        with open(base + ".json", "w") as fh:
            fh.write(e.song.params.to_json())
        with open(base + ".txt", "w") as fh:
            fh.write(plan_text(e, i))
        if audio:
            e.files["audio"] = preview(e.song, base)
        if progress:
            progress(f"  wrote {os.path.basename(base)}")

    path = os.path.join(out_dir, "setlist.md")
    with open(path, "w") as fh:
        fh.write(setlist_markdown(entries))
    with open(os.path.join(out_dir, "setlist.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["#", "title", "profile", "bpm", "key", "camelot", "minutes",
                    "hook", "score", "listener", "critic", "seed", "params"])
        for i, e in enumerate(entries, 1):
            s = e.song
            w.writerow([i, e.title, s.params.profile, s.bpm, str(s.scale), camelot(s),
                        f"{e.minutes:.2f}", _hook_label(s), f"{e.score:.3f}",
                        f"{e.listener:.3f}", f"{e.critic:.2f}", s.seed,
                        os.path.basename(e.files.get("json", ""))])
    return path


def preview(song: Song, base: str) -> str:
    """Render an audio preview next to the song's files. AAC (.m4a) through
    macOS's own `afconvert` when it is there - a fifth the size of the WAV,
    which matters across a whole set - else the WAV itself."""
    import shutil
    import subprocess

    from .render import audio as audio_mod
    wav = audio_mod.write(song, base + ".wav")
    if shutil.which("afconvert"):
        m4a = base + ".m4a"
        done = subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "192000",
                               wav, m4a], capture_output=True)
        if done.returncode == 0 and os.path.exists(m4a):
            os.remove(wav)
            return m4a
    return wav


def _hook_label(song: Song) -> str:
    h = song.params.hook
    return f"{h.form}/{song.params.lead_style}" if h else "-"


def plan_text(e: Entry, index: int) -> str:
    s = e.song
    lines = [
        f"{index:02d}  {e.title}",
        f"{s.params.profile}  {s.bpm}bpm  {s.scale}  ({camelot(s)})  "
        f"{e.minutes:.2f} min  seed {s.seed}",
        f"score {e.score:.3f}  (listener {e.listener:.3f}, critic {e.critic:.2f})",
        "",
        s.summary(),
        "",
        s.arrangement.describe(s.bpm),
        "",
        "Listener: " + analyze.analyze(s).line(),
        "",
        "Performance notes:",
    ]
    lines += [f"  - {n}" for n in s.notes_for_player]
    lines += [
        "",
        "To put this exact song on the Deluge:",
        f"  ./riddim-cli send --params {os.path.basename(e.files.get('json', 'this.json')) or 'this.json'}",
        "",
    ]
    return "\n".join(lines)


def setlist_markdown(entries: list[Entry]) -> str:
    total = sum(e.minutes for e in entries)
    lines = [
        "# Set list",
        "",
        f"{len(entries)} tracks, {int(total // 60)}h{int(total % 60):02d}m "
        f"({total:.0f} minutes). Ordered as a set: tempo blocks from roots to "
        "140, keys chained a step at a time round the circle of fifths.",
        "",
        "| # | title | feel | bpm | key | camelot | min | hook | score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    prev = None
    for i, e in enumerate(entries, 1):
        s = e.song
        mark = ""
        if prev is not None and prev.params.profile != s.params.profile:
            mark = " ⟶"
        lines.append(
            f"| {i}{mark} | {e.title} | {s.params.profile} | {s.bpm} | {s.scale} | "
            f"{camelot(s)} | {e.minutes:.1f} | {_hook_label(s)} | {e.score:.2f} |")
        prev = s
    lines += [
        "",
        "⟶ marks a tempo change. Going from a 70s dub block into 140 dubstep is",
        "a half-time move: 140 is felt at 70, so the groove carries straight over.",
        "",
        "Each track's `.json` reproduces it exactly: `./riddim-cli send --params NN_title.json`.",
    ]
    return "\n".join(lines) + "\n"


def build(
    count: int,
    profiles=SET_ORDER,
    candidates: int = 6,
    seed: int = 1,
    fix: bool = True,
    rig: Rig | None = None,
    progress=None,
) -> list[Entry]:
    """Compose `count` songs spread across `profiles`, best of `candidates`
    seeds each, ordered as a set."""
    profiles = list(profiles)
    entries = []
    for i in range(count):
        prof = profiles[i % len(profiles)]
        first = seed + i * candidates
        e = best_of(prof, range(first, first + candidates), rig=rig, fix=fix)
        entries.append(e)
        if progress:
            progress(f"[{i + 1}/{count}] {prof:<15} {e.title:<24} {e.song.bpm}bpm "
                     f"{str(e.song.scale):<18} {e.minutes:.1f}min  score {e.score:.3f}")
    _unique_titles(entries)
    return order(entries)


def _unique_titles(entries: list[Entry]) -> None:
    """A set list cannot have two songs with one name: redraw the noun."""
    seen = set()
    for e in entries:
        rng = random.Random(f"retitle:{e.song.seed}")
        while e.title in seen:
            e.title = f"{e.title.split()[0]} {rng.choice(_NOUN)}"
        seen.add(e.title)


# What a variation can change, from smallest to largest move. Twists keep
# every seed; rolls redraw one seed and keep everything else.
VARY = {
    "mutate": ("twist", ("drums", "bass", "lead", "hook", "interplay", "dynamics", "chords"), 0.5),
    "hook": ("roll", "hook", 0.0),
    "groove": ("twist", ("drums", "bass", "chords"), 0.6),
    "arrangement": ("twist", ("interplay", "dynamics"), 0.7),
    "structure": ("roll", "structure", 0.0),
    "drums": ("roll", "drums", 0.0),
    "bass": ("roll", "sub", 0.0),
    "wild": ("twist", ("drums", "bass", "lead", "hook", "interplay", "dynamics", "chords",
                       "harmony"), 1.0),
}


def variations(song: Song, count: int, how: str = "mutate", seed: int = 1,
               rig: Rig | None = None) -> list[Entry]:
    """`count` variations of a song, each judged, best first.

    `how` is a key of VARY: a twist (knob mutation with every seed frozen -
    the same song moved) or a roll (one seed redrawn - a new hook, a new
    drum take, a new structure - over everything else unchanged).
    """
    from .compose import regenerate
    from .direct import apply_ops, mutation

    if how not in VARY:
        raise ValueError(f"unknown variation {how!r}; known: {', '.join(VARY)}")
    kind, what, amount = VARY[how]
    out = []
    for i in range(count):
        rng = random.Random(f"{song.seed}:{how}:{seed}:{i}")
        if kind == "twist":
            ops = mutation(song.params, what, amount, rng)
            v = compose(params=apply_ops(song.params, ops), rig=rig)
        else:
            v = regenerate(song, what, rig=rig, rng=rng)
        combined, listener, crit = judge(v)
        title = f"{title_for(song)} ({how} {i + 1})"
        out.append(Entry(v, title, combined, listener, crit))
    out.sort(key=lambda e: -e.score)
    return out


def load(path: str, rig: Rig | None = None) -> Song:
    """A song back from its params file - byte-identical to when it was written."""
    with open(path) as fh:
        params = SongParams.from_json(fh.read())
    return compose(params=params, rig=rig)

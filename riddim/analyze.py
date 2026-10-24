"""Measure a finished song the way a listener hears it.

The critic (`critique.py`) asks "is anything broken?". This asks "is it any
good?" - and answers in numbers, so a batch of songs can be ranked and a
change to the engine can be shown to help rather than hoped to. Every
measure is a plain property of the notes:

- **hook**       the hook's own score: shape, return, climax, close
- **fit**        lead notes that agree with the chord under them, where it
                 counts (strong beats and held notes)
- **clashes**    lead notes a semitone from a chord or pad note sounding at
                 the same moment - the sound of two parts not listening
- **register**   the lead above the stabs sounding at the same moment, not
                 inside them
- **space**      chord stabs that do not land on the lead's notes
- **answers**    the lead's breaths that somebody fills - the conversation
- **contrast**   how much louder and busier the peaks are than the holes
- **arc**        sections' note density and velocity following their energy
- **tension**    builds that roll, drops that land on silence and a crash
- **length**     a playable duration: three and a half to six minutes
- **presence**   the hook actually heard in the peaks, by lead or bass

`score` folds them into one 0..1 number with readable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .arrange import structure
from .theory import harmony
from .song import Song
from .units import TICKS_PER_16TH, TICKS_PER_BAR

CRASH, RISER, SNARE = 42, 43, 37

WEIGHTS = {
    "hook": 2.0, "fit": 1.5, "clashes": 1.5, "register": 0.8, "space": 0.7,
    "answers": 0.8, "contrast": 1.2, "arc": 1.0, "tension": 1.2,
    "length": 0.8, "presence": 1.2,
}


@dataclass
class Report:
    metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def score(self) -> float:
        total = sum(WEIGHTS[k] * max(0.0, min(1.0, v)) for k, v in self.metrics.items()
                    if k in WEIGHTS)
        return round(total / sum(WEIGHTS[k] for k in self.metrics if k in WEIGHTS), 4)

    def line(self) -> str:
        return " ".join(f"{k}={v:.2f}" for k, v in self.metrics.items())


def _sounding(notes, tick):
    return [n for n in notes if n.start <= tick < n.end]


def analyze(song: Song) -> Report:
    r = Report()
    m = r.metrics
    secs = song.arrangement.sections
    lead = song.part("lead")
    chords = song.part("chords")
    pad = song.part("pad")
    sub = song.part("sub")
    drums = song.part("drums")

    concept = song.concept or ""
    hook_score = None
    if "score=" in concept:
        try:
            hook_score = float(concept.split("score=")[1].split()[0])
        except ValueError:
            hook_score = None
    m["hook"] = hook_score if hook_score is not None else 0.5

    # Fit and clashes.
    if lead is not None and lead.notes and song.harmony is not None:
        counted = agree = 0
        clash = 0
        harmony_notes = (chords.notes if chords else []) + (pad.notes if pad else [])
        for n in lead.notes:
            pos = (n.start % TICKS_PER_BAR) // TICKS_PER_16TH
            heavy = pos % 4 == 0 or n.length >= 2 * TICKS_PER_16TH
            ev = song.harmony.at(n.start)
            if heavy and ev is not None:
                counted += 1
                # The root counts: the voicing leaves it out because the
                # bass is playing it, not because it is not in the chord.
                # So do the chord's available tensions - the ninth, the minor
                # eleventh, the sixth, the seventh: colour, not a wrong note.
                if harmony.sits_on(n.midi, ev.chord):
                    agree += 1
            if heavy:
                for h in _sounding(harmony_notes, n.start + 1):
                    if (n.midi - h.midi) % 12 in (1, 11):
                        clash += 1
                        break
        m["fit"] = agree / counted if counted else 1.0
        m["clashes"] = 1.0 - min(1.0, 4.0 * clash / max(1, len(lead.notes)))
        if chords is not None and chords.notes:
            # Above the stabs *sounding at that moment*: a lead note below
            # the chords' song-wide top while no stab plays is not inside
            # them - only one under a sounding stab is masked.
            under = 0
            for n in lead.notes:
                if any(c.midi >= n.midi for c in _sounding(chords.notes, n.start + 1)):
                    under += 1
            m["register"] = 1.0 - min(1.0, 4.0 * under / len(lead.notes))
            lead_on = {n.start // TICKS_PER_16TH for n in lead.notes}
            stabs = {n.start // TICKS_PER_16TH for n in chords.notes}
            both = [s for s in stabs if s in lead_on]
            m["space"] = 1.0 - len(both) / max(1, len(stabs))
        else:
            m["register"] = 1.0
            m["space"] = 1.0
        # Answers: lead gaps of a beat or more inside sections where the lead
        # sings, and whether another pitched part starts a note in them.
        others = [p for p in song.parts if p.role in ("chords", "sub", "pad")]
        starts = sorted({n.start for p in others for n in p.notes})
        gaps = filled = 0
        ln = lead.sorted_notes()
        for a, b in zip(ln, ln[1:]):
            if b.start - a.end >= 4 * TICKS_PER_16TH and b.start - a.end <= 2 * TICKS_PER_BAR:
                gaps += 1
                if any(a.end <= s < b.start for s in _slice(starts, a.end, b.start)):
                    filled += 1
        m["answers"] = filled / gaps if gaps else 0.5
    else:
        m["fit"] = m["clashes"] = 0.5
        m["register"] = m["space"] = m["answers"] = 0.5

    # Contrast and arc.
    dens, vels, energies = [], [], []
    for s in secs:
        n = sum(len(p.in_bars(s.start_bar, s.end_bar)) for p in song.parts)
        vs = [x.velocity for p in song.parts for x in p.in_bars(s.start_bar, s.end_bar)]
        dens.append(n / max(1, s.bars))
        vels.append(sum(vs) / len(vs) if vs else 0)
        energies.append(s.energy)
    peaks = [d for d, s in zip(dens, secs) if structure.kind_of(s) == "peak"]
    holes = [d for d, s in zip(dens, secs) if structure.kind_of(s) in ("release", "intro", "outro")]
    if peaks and holes:
        ratio = (sum(peaks) / len(peaks)) / max(1e-6, sum(holes) / len(holes))
        m["contrast"] = min(1.0, max(0.0, (ratio - 1.0) / 2.0))
    else:
        m["contrast"] = 0.4
    m["arc"] = max(0.0, _corr(energies, dens) * 0.6 + _corr(energies, vels) * 0.4)

    # Tension: builds into drops roll; drops land after silence on a crash.
    drops = scored = 0
    for prev, s in zip(secs, secs[1:]):
        if structure.kind_of(s) != "peak" or s.energy - prev.energy < 0.25:
            continue
        drops += 1
        pts = 0.0
        if drums is not None:
            tail = [n for n in drums.notes if n.midi == SNARE and
                    prev.start_tick + max(0, prev.bars - 2) * TICKS_PER_BAR <= n.start < s.start_tick]
            pts += 0.35 if len(tail) >= 10 else (0.15 if len(tail) >= 4 else 0.0)
            if any(n.midi == CRASH and n.start == s.start_tick for n in drums.notes):
                pts += 0.3
        window = (s.start_tick - TICKS_PER_BAR // 8, s.start_tick)
        loud = [n for p in song.parts for n in p.notes
                if window[0] <= n.start < window[1] and n.midi != RISER]
        pts += 0.35 if not loud else 0.0
        scored += pts
    m["tension"] = scored / drops if drops else 0.5

    minutes = song.arrangement.duration_seconds(song.bpm) / 60.0
    # A dub version is a record side - three to four and a half minutes on
    # the references; a club record runs longer.
    family = ""
    if song.params is not None:
        from . import profile as _profile
        try:
            family = _profile.get(song.params.profile).family
        except (KeyError, ValueError):
            family = ""
    lo, hi, mid = (2.9, 5.0, 3.8) if family == "dub" else (3.4, 6.5, 4.5)
    m["length"] = 1.0 if lo <= minutes <= hi else max(0.0, 1.0 - abs(minutes - mid) / 4.0)

    # Presence: the hook heard in peaks, by the lead or by the bass taking it.
    pk = [s for s in secs if structure.kind_of(s) == "peak"]
    if pk:
        heard = 0
        for s in pk:
            # Bars in which the lead is sounding - a held melodica note is as
            # present as a riff, so this counts time, not attacks.
            sounding = set()
            for n in (lead.notes if lead else []):
                if n.end > s.start_tick and n.start < s.start_tick + s.length_ticks:
                    for b in range(n.start // TICKS_PER_BAR, (n.end - 1) // TICKS_PER_BAR + 1):
                        if s.start_bar <= b < s.end_bar:
                            sounding.add(b)
            if len(sounding) >= 0.5 * s.bars:
                heard += 1
            elif sub is not None and s.plays("sub"):
                heard += 0.5
        m["presence"] = heard / len(pk)
    else:
        m["presence"] = 0.5
    return r


def _slice(sorted_starts, a, b):
    import bisect
    i = bisect.bisect_left(sorted_starts, a)
    j = bisect.bisect_left(sorted_starts, b)
    return sorted_starts[i:j]


def _corr(xs, ys) -> float:
    n = len(xs)
    if n < 3:
        return 0.5
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def score(song: Song) -> float:
    return analyze(song).score

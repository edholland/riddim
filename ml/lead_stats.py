import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from riddim.theory import harmony
from riddim.units import TICKS_PER_BAR, TICKS_PER_16TH

def stats(notes, song, t0, t1):
    ns = [n for n in notes if t0 <= n.start < t1]
    bars = (t1 - t0) / TICKS_PER_BAR
    sc = set(song.scale.pitch_classes)
    in_key = sum(n.midi % 12 in sc for n in ns) / max(1, len(ns))
    fit = []
    for n in ns:
        ev = song.harmony.at(n.start)
        if ev is not None and ((n.start % TICKS_PER_BAR) // TICKS_PER_16TH) % 4 == 0:
            fit.append(harmony.sits_on(n.midi, ev.chord))
    iv = [abs(b.midi - a.midi) for a, b in zip(ns, ns[1:])]
    return dict(per_bar=round(len(ns) / bars, 1), in_key=round(in_key, 2),
                strong_fit=round(sum(fit) / max(1, len(fit)), 2),
                range=(min(n.midi for n in ns), max(n.midi for n in ns)) if ns else None,
                mean_leap=round(sum(iv) / max(1, len(iv)), 1),
                repeats=round(sum(1 for x in iv if x == 0) / max(1, len(iv)), 2))

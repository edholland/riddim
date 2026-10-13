"""A/B: riddim's own lead vs a lead written by the Anticipatory Music
Transformer over riddim's band (drums, sub, chords, pad as controls).

For the first drop of a song: the 8 bars before it are context, the drop is
written. The model may only play one instrument (a saw lead) inside the
register the original lead used. Its notes are snapped to the sixteenth
grid and made monophonic, then put back into the song in place of the
original lead for that drop. Writes <out>/<name>_A.wav (riddim) and _B.wav
(model), plus the model's lead as MIDI.

usage: amt_lead.py PROFILE SEED [TAKES] [MODEL_DIR]
"""
import os
import sys
import time
from dataclasses import replace


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from anticipation import ops                          # noqa: E402
from anticipation.config import TIME_RESOLUTION       # noqa: E402
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET  # noqa: E402
from transformers import AutoModelForCausalLM         # noqa: E402

from lead_core import LEAD_INSTR, write_lead          # noqa: E402
from riddim.compose import compose                   # noqa: E402
from riddim.render import audio                       # noqa: E402
from riddim.render.midifile import GM_DRUMS           # noqa: E402
from riddim.song import Note                          # noqa: E402
from riddim.units import TICKS_PER_16TH, TICKS_PER_BAR  # noqa: E402

PROGRAMS = {"sub": 38, "chords": 4, "pad": 89}
MAX_SECONDS = 95.0
ML_HOME = os.environ.get("ML_HOME", os.path.expanduser("~/ml"))


def band_events(song, t0, t1, sec_per_tick):
    """riddim's band in [t0, t1) ticks as AMT event tokens, from time 0."""
    ev = []
    for part in song.parts:
        if part.role == "lead":
            continue
        drums = part.output == "kit"
        instr = 128 if drums else PROGRAMS.get(part.role, 0)
        for n in part.notes:
            if not t0 <= n.start < t1:
                continue
            pitch = GM_DRUMS.get(n.midi, n.midi) if drums else n.midi
            t = int(round((n.start - t0) * sec_per_tick * TIME_RESOLUTION))
            d = max(1, min(999, int(round(n.length * sec_per_tick * TIME_RESOLUTION))))
            ev += [TIME_OFFSET + t, DUR_OFFSET + d, NOTE_OFFSET + instr * 128 + pitch]
    return ops.sort(ev)


def to_grid(notes, t0_tick, sec_per_tick, start_s, end_tick):
    """Seconds -> riddim notes on the sixteenth grid, monophonic, accented."""
    placed = {}
    for s, d, p in notes:
        if s < start_s:
            continue
        step = int(round(s / sec_per_tick / TICKS_PER_16TH))
        length = max(1, int(round(d / sec_per_tick / TICKS_PER_16TH)))
        tick = t0_tick + step * TICKS_PER_16TH
        if tick >= end_tick:
            continue
        placed.setdefault(tick, (p, length))
    out = []
    ticks = sorted(placed)
    for i, tick in enumerate(ticks):
        p, length = placed[tick]
        nxt = ticks[i + 1] if i + 1 < len(ticks) else end_tick
        dur = min(length * TICKS_PER_16TH, nxt - tick, end_tick - tick)
        strong = (tick % TICKS_PER_BAR) % (4 * TICKS_PER_16TH) == 0
        out.append(Note(p, tick, max(TICKS_PER_16TH // 2, dur), 100 if strong else 86))
    return out


def main():
    profile, seed = sys.argv[1], int(sys.argv[2])
    takes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(ML_HOME, "models", "music-medium-800k")
    out_dir = os.path.join(ML_HOME, "out", "ab")
    tag = "model" if model_dir.rstrip("/").endswith("music-medium-800k") else os.path.basename(model_dir.rstrip("/"))
    os.makedirs(out_dir, exist_ok=True)

    song = compose(profile=profile, seed=seed)
    sec_per_tick = 60.0 / song.bpm / 96
    secs = song.arrangement.sections
    drop = next(s for s in secs if s.energy >= 0.8)
    t0 = max(0, drop.start_bar - 8) * TICKS_PER_BAR
    t1 = drop.start_tick + drop.length_ticks
    while (t1 - t0) * sec_per_tick > MAX_SECONDS and t1 - TICKS_PER_BAR * 8 > drop.start_tick:
        t1 -= TICKS_PER_BAR * 8
    start_s = (drop.start_tick - t0) * sec_per_tick
    end_s = (t1 - t0) * sec_per_tick

    lead = song.part("lead")
    in_drop = [n.midi for n in lead.notes if drop.start_tick <= n.start < t1] if lead else []
    lo, hi = (min(in_drop) - 2, max(in_drop) + 2) if in_drop else (64, 88)

    model = AutoModelForCausalLM.from_pretrained(model_dir).to("mps").eval()
    controls = band_events(song, t0, t1, sec_per_tick)
    # the original lead's 8 bars of build as context, so the model hears the tune's world
    for take in range(takes):
        t = time.time()
        eighth = int(round(TICKS_PER_16TH * 2 * sec_per_tick * TIME_RESOLUTION))
        notes = write_lead(model, controls, start_s, end_s, lo, hi, 0.9, seed * 100 + take,
                           set(song.scale.pitch_classes), eighth)
        grid = to_grid(notes, t0, sec_per_tick, start_s, t1)
        import lead_stats
        if take == 0:
            print("riddim lead:", lead_stats.stats(lead.notes, song, drop.start_tick, t1))
        print("model lead: ", lead_stats.stats(grid, song, drop.start_tick, t1))
        print(f"take {take}: {len(grid)} notes over {(t1 - drop.start_tick) // TICKS_PER_BAR} bars "
              f"(range {lo}-{hi}) in {time.time() - t:.0f}s")
        b = compose(params=song.params)
        bl = b.part("lead")
        bl.notes = [n for n in bl.notes if not drop.start_tick <= n.start < t1] + grid
        bl.notes.sort(key=lambda n: n.start)
        name = f"{profile}_{seed}"
        trim = lambda s: _window(s, t0, t1)  # noqa: E731
        if take == 0:
            audio.write(trim(song), f"{out_dir}/{name}_A_riddim.wav")
        audio.write(trim(b), f"{out_dir}/{name}_B_{tag}_take{take}.wav")


def _window(song, t0, t1):
    """The song cut to [t0, t1): the build into the drop and the drop."""
    s = compose(params=song.params)
    s.parts = []
    for part in song.parts:
        p = replace(part, notes=[replace(n, start=n.start - t0) for n in part.notes if t0 <= n.start < t1])
        s.parts.append(p)
    s.coda = []
    bars = (t1 - t0) // TICKS_PER_BAR
    s.arrangement.sections = [replace(sec, start_bar=0, bars=bars) for sec in s.arrangement.sections[:1]]
    return s


if __name__ == "__main__":
    main()

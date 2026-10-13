"""Compare models as lead writers, on riddim songs, the way riddim uses them.

For each model and each song: write the drops' lead through riddim's own
model_lead (hook drafts, chosen, arranged AABA), then score it - the hook's
phrase score, how often it lands on the chord on the beat, how much it
moves, how much it repeats - and render the winners for listening.

usage: eval_leads.py OUT_DIR MODEL_DIR [MODEL_DIR ...] [--songs 12] [--style rhythm]
                     [--audio N]
Writes OUT_DIR/report.json, OUT_DIR/report.md and (with --audio) the first N
songs as .m4a per model next to riddim's own lead.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from riddim import model_lead as ml               # noqa: E402
from riddim.compose import compose                # noqa: E402
from riddim.theory import harmony                 # noqa: E402
from riddim.units import TICKS_PER_BAR, TICKS_PER_BEAT  # noqa: E402

PROFILES = ["dubstep_140", "dark_garage", "uk_garage", "dubby_halftime", "steppers_dub", "roots_dub"]


def stats(song, which):
    notes = [n for sec in ml.drops(song) for n in song.part("lead").notes
             if sec.start_tick <= n.start < sec.start_tick + sec.length_ticks]
    if len(notes) < 2:
        return None
    first = ml.drops(song)[0]
    phrase = [n for n in notes if first.start_tick <= n.start < first.start_tick + ml.PHRASE_BARS * TICKS_PER_BAR]
    strong = [harmony.sits_on(n.midi, ev.chord) for n in notes
              if n.start % TICKS_PER_BEAT == 0 and (ev := song.harmony.at(n.start)) is not None]
    leaps = [abs(b.midi - a.midi) for a, b in zip(notes, notes[1:])]
    bars = sum(s.bars for s in ml.drops(song))
    return {
        "hook_score": round(ml.phrase_score(phrase, song, first.start_tick, ml.PHRASE_BARS), 3),
        "strong_fit": round(sum(strong) / max(1, len(strong)), 3),
        "per_bar": round(len(notes) / bars, 2),
        "mean_leap": round(sum(leaps) / len(leaps), 2),
        "repeats": round(sum(1 for x in leaps if x == 0) / len(leaps), 3),
        "distinct": len({n.midi for n in notes}),
    }


def render(song, path):
    from riddim.render import audio
    wav = path + ".wav"
    audio.write(song, wav)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "192000", wav, path + ".m4a"], check=True)
    os.unlink(wav)


def main():
    out = sys.argv[1]

    def opt(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    flags = {"--songs", "--style", "--audio"}
    models = [a for i, a in enumerate(sys.argv[2:], 2)
              if not a.startswith("--") and sys.argv[i - 1] not in flags]
    n_songs, style, n_audio = int(opt("--songs", 12)), opt("--style", "rhythm"), int(opt("--audio", 0))
    os.makedirs(out, exist_ok=True)
    songs = [compose(profile=PROFILES[i % len(PROFILES)], seed=500 + i, live=True) for i in range(n_songs)]
    report = {"riddim": [stats(s, "riddim") for s in songs]}
    for i, s in enumerate(songs[:n_audio]):
        render(s, os.path.join(out, f"{i:02d}_{s.params.profile}_A_riddim"))
    for m in models:
        name = os.path.basename(os.path.dirname(m.rstrip("/"))) + "_" + os.path.basename(m.rstrip("/"))
        t = time.time()
        with ml.Writer(model=m) as w:
            rows = []
            for i, s in enumerate(songs):
                lead = ml.write(s, w, style=style)
                rows.append(stats(lead, name))
                if i < n_audio:
                    render(lead, os.path.join(out, f"{i:02d}_{s.params.profile}_{name}"))
        report[name] = rows
        print(f"{name}: {len(songs)} songs in {time.time() - t:.0f}s", flush=True)
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
    keys = ["hook_score", "strong_fit", "per_bar", "mean_leap", "repeats", "distinct"]
    lines = [f"# Lead writers on {n_songs} riddim live tracks ({style})", "",
             "| writer | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, rows in report.items():
        rows = [r for r in rows if r]
        avg = {k: sum(r[k] for r in rows) / len(rows) for k in keys} if rows else {}
        lines.append(f"| {name} | " + " | ".join(f"{avg.get(k, 0):.2f}" for k in keys) + " |")
    open(os.path.join(out, "report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

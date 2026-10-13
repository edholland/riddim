"""Score every MIDI file for the 2006 DMZ sound by what it plays.

tempo   dub (65-95) or dubstep (128-145, felt at half)
drums   half-time (snare on beat 3 at ~140), one-drop (kick+snare on 3,
        beat 1 empty), steppers (four kicks) - not a 2-and-4 rock beat
skank   chord hits on the offbeats
bass    a bass part that carries the low end, sparse rather than walking
mode    minor / modal pitch-class profile

Writes one JSON line per file: path + features + score.
"""
import json
import os
import sys
from multiprocessing import Pool

import mido

KICK = {35, 36}
SNARE = {37, 38, 39, 40}
# Krumhansl-Kessler key profiles
MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def corr(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def features(path):
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    tpb = mid.ticks_per_beat or 480
    tempo = None
    notes = []          # (abs_tick, channel, pitch, velocity, program)
    for track in mid.tracks:
        t = 0
        prog = {}
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo" and tempo is None:
                tempo = msg.tempo
            elif msg.type == "program_change":
                prog[msg.channel] = msg.program
            elif msg.type == "note_on" and msg.velocity > 0:
                notes.append((t, msg.channel, msg.note, msg.velocity, prog.get(msg.channel, 0)))
    if not notes or tempo is None:
        return None
    bpm = 60_000_000 / tempo
    end = max(n[0] for n in notes)
    bars = end / (4 * tpb)
    if bars < 16:
        return None
    step = tpb / 4      # a sixteenth

    def pos16(tick):
        return int(round((tick % (4 * tpb)) / step)) % 16

    drums = [n for n in notes if n[1] == 9]
    pitched = [n for n in notes if n[1] != 9]
    f = {"bpm": round(bpm, 1), "bars": round(bars), "n": len(notes)}

    # Drums: where kick and snare land in the bar
    kick = [0] * 16
    snare = [0] * 16
    for n in drums:
        p = pos16(n[0])
        if n[2] in KICK:
            kick[p] += 1
        elif n[2] in SNARE:
            snare[p] += 1
    ks, ss = sum(kick) or 1, sum(snare) or 1
    f["snare_on3"] = round(snare[8] / ss, 2)                  # half-time / one-drop backbeat
    f["snare_24"] = round((snare[4] + snare[12]) / ss, 2)      # rock / straight backbeat
    f["kick_on1"] = round(kick[0] / ks, 2)
    f["kick_on3"] = round(kick[8] / ks, 2)
    f["kick_4"] = round(sum(kick[i] for i in (0, 4, 8, 12)) / ks, 2)
    f["has_drums"] = len(drums) > 4 * bars

    # Bass: GM bass programs 32-39, or the low register
    bass = [n for n in pitched if 32 <= n[4] <= 39 or n[2] < 48]
    f["bass_per_bar"] = round(len(bass) / bars, 2)
    f["has_bass"] = len(bass) > 2 * bars

    # Skank: chord onsets (3+ pitched notes at one tick, above the bass) on offbeats
    by_tick = {}
    for n in pitched:
        if n[2] >= 52:
            by_tick.setdefault(n[0], 0)
            by_tick[n[0]] += 1
    chords = [t for t, c in by_tick.items() if c >= 3]
    if chords:
        off = sum(1 for t in chords if pos16(t) % 4 == 2)
        f["skank"] = round(off / len(chords), 2)
    else:
        f["skank"] = 0.0
    f["chords_per_bar"] = round(len(chords) / bars, 2)

    # Mode: best minor vs major correlation over all 12 keys
    pc = [0.0] * 12
    for n in pitched:
        pc[n[2] % 12] += 1
    best_min = max(corr(pc, MIN[-k:] + MIN[:-k]) for k in range(12))
    best_maj = max(corr(pc, MAJ[-k:] + MAJ[:-k]) for k in range(12))
    f["minorness"] = round(best_min - best_maj, 3)
    f["density"] = round(len(pitched) / bars, 1)

    # The score
    s = 0.0
    dub_tempo = 62 <= bpm <= 95
    step_tempo = 128 <= bpm <= 146 or 64 <= bpm <= 73     # 140 written or felt
    if dub_tempo or step_tempo:
        s += 1.0
    if f["has_drums"]:
        if f["snare_on3"] > 0.45 and f["snare_24"] < 0.35:
            s += 2.0                                          # half-time or one-drop backbeat
        if f["kick_on3"] > 0.3 and f["kick_on1"] < 0.15:
            s += 1.0                                          # the one-drop
        if f["kick_4"] > 0.8 and dub_tempo:
            s += 0.5                                          # steppers
        if f["snare_24"] > 0.6:
            s -= 1.5                                          # a rock/pop backbeat
    else:
        s -= 1.0
    if f["has_bass"]:
        s += 0.5
        if f["bass_per_bar"] <= 6:
            s += 0.5                                          # sub-led, not walking eighths
    s += 1.5 * min(1.0, f["skank"] * 1.5) if f["chords_per_bar"] >= 0.5 else 0.0
    s += 1.0 if f["minorness"] > 0.05 else (-0.5 if f["minorness"] < -0.1 else 0.0)
    if f["density"] > 60:
        s -= 1.0                                              # too busy for the space
    f["score"] = round(s, 2)
    return f


def work(path):
    f = features(path)
    if f is None:
        return None
    f["path"] = path
    return f


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    paths = [os.path.join(d, x) for d, _, fs in os.walk(root) for x in fs if x.endswith(".mid")]
    print(len(paths), "files", flush=True)
    done = 0
    with Pool(16) as pool, open(out, "w") as fh:
        for f in pool.imap_unordered(work, paths, chunksize=64):
            done += 1
            if f:
                fh.write(json.dumps(f) + "\n")
            if done % 10000 == 0:
                print(done, flush=True)
    print("DONE", flush=True)

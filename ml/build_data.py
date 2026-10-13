"""Build the training data for the lead writer from open MIDI.

The model is asked one thing at inference: here is riddim's band (sub on
GM 38, chords on 4, pad on 89, drums), at 66-142 bpm, write the lead (GM 81).
So the training data says exactly that, in exactly those instruments:

1. role-normalise every song - one lead (the melody track: the most
   monophonic, highest, most present part, or one named melody/vocal/lead),
   one bass, up to two comping parts (short notes -> chords, held -> pad),
   drums; everything else dropped, the way riddim's band is sparse;
2. move it to a riddim tempo - time is rescaled so the song's beat lands on
   the nearest riddim tempo (dub 66-94, garage/dubstep 130-142), never by
   more than a quarter;
3. transpose it (a random shift, so every key is seen);
4. tokenize two ways:
   * lead:  the song up to a split as context, then the band as anticipated
            controls and the lead as the events - the inference task itself;
   * full:  the Anticipatory Music Transformer's own augmented tokenization,
            on the normalised songs, so the model keeps its general skill.

usage: build_data.py SCORES.jsonl OUT_DIR [--min-score 3] [--max-score 99] [--variants 3]
                     [--full-augment 2] [--limit N] [--jobs 16]
SCORES.jsonl: dmz_filter.py output (path + score per file).
Writes OUT_DIR/{lead,full}/train-XX.npy + valid.npy and OUT_DIR/stats.json.
"""
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import mido
import numpy as np

from anticipation import ops
from anticipation.config import CONTEXT_SIZE, DELTA, EVENT_SIZE, M, MAX_TIME, TIME_RESOLUTION
from anticipation.tokenize import extract_instruments, extract_random, extract_spans
from anticipation.vocab import (ANTICIPATE, AUTOREGRESS, CONTROL_OFFSET, DUR_OFFSET, MAX_DUR,
                                NOTE_OFFSET, SEPARATOR, TIME_OFFSET)

LEAD, BASS, CHORDS, PAD, DRUMS = 81, 38, 4, 89, 128
TEMPOS = [66, 70, 74, 78, 82, 86, 90, 94, 130, 133, 136, 138, 140, 142]
MAX_STRETCH = 1.25
LEADISH = set(range(52, 88)) - set(range(88, 96))       # voices, brass, reeds, pipes, synth leads
NAMES = ("melody", "vocal", "voice", "lead", "solo", "melodie", "melodía", "sing")


# -- reading -------------------------------------------------------------------

def read(path):
    """Notes per channel as (start_s, dur_s, pitch), programs, track names, first bpm."""
    mid = mido.MidiFile(path)
    names = {}
    for tr in mid.tracks:
        nm = ""
        chans = set()
        for msg in tr:
            if msg.type == "track_name":
                nm = msg.name
            elif hasattr(msg, "channel"):
                chans.add(msg.channel)
        for c in chans:
            names.setdefault(c, nm.lower())
    notes = defaultdict(list)
    open_ = defaultdict(list)
    prog = defaultdict(int)
    progs = {}
    bpm = None
    t = 0.0
    for msg in mid:                      # merged, times in seconds
        t += msg.time
        if msg.type == "set_tempo" and bpm is None:
            bpm = 60e6 / msg.tempo
        elif msg.type == "program_change":
            prog[msg.channel] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            open_[(msg.channel, msg.note)].append(t)
            progs.setdefault(msg.channel, prog[msg.channel])
        elif msg.type in ("note_off", "note_on"):
            q = open_.get((msg.channel, msg.note))
            if q:
                s = q.pop(0)
                notes[msg.channel].append((s, t - s, msg.note))
    return notes, progs, names, bpm or 120.0


def _poly(ns):
    """Fraction of notes that start while another note in the part still sounds."""
    ns = sorted(ns)
    over, end = 0, -1.0
    for s, d, _ in ns:
        if s < end - 0.02:
            over += 1
        end = max(end, s + d)
    return over / max(1, len(ns))


def roles(notes, progs, names, beat):
    """channel -> GM program in riddim's band, or None to drop the part."""
    if not notes:
        return {}
    end = max(s + d for ns in notes.values() for s, d, _ in ns)
    bars = max(1.0, end / (4 * beat))
    info = {}
    for ch, ns in notes.items():
        if ch == 9 or len(ns) < 8:
            continue
        pitches = sorted(p for _, _, p in ns)
        bars_hit = len({int(s / (4 * beat)) for s, _, _ in ns})
        info[ch] = dict(
            n=len(ns), median=pitches[len(pitches) // 2], poly=_poly(ns),
            cover=bars_hit / bars, dur=sum(d for _, d, _ in ns) / len(ns) / beat,
            prog=progs.get(ch, 0), name=names.get(ch, ""))
    out = {9: DRUMS} if 9 in notes and len(notes[9]) >= 16 else {}
    # the lead
    best, best_score = None, 0.0
    for ch, f in info.items():
        if f["poly"] > 0.25 or f["median"] < 58 or f["cover"] < 0.2 or f["n"] / bars < 0.5:
            continue
        sc = f["cover"] * (1 - f["poly"]) + 0.01 * (f["median"] - 60)
        if f["prog"] in LEADISH:
            sc += 0.3
        if any(k in f["name"] for k in NAMES):
            sc += 0.8
        if sc > best_score:
            best, best_score = ch, sc
    if best is None:
        return {}
    out[best] = LEAD
    # the bass
    lows = [(f["n"], ch) for ch, f in info.items()
            if ch != best and (f["median"] < 52 or 32 <= f["prog"] <= 39) and f["poly"] < 0.4]
    if lows:
        out[max(lows)[1]] = BASS
    # up to two comping parts
    comp = sorted(((f["n"], ch) for ch, f in info.items()
                   if ch not in out and f["poly"] >= 0.3 and f["median"] >= 48), reverse=True)
    for _, ch in comp[:2]:
        out[ch] = PAD if info[ch]["dur"] >= 1.0 else CHORDS
    return out


def target_tempo(bpm):
    best = min(TEMPOS, key=lambda t: abs(np.log(t / bpm)))
    f = bpm / best
    return best if 1 / MAX_STRETCH <= f <= MAX_STRETCH else None


def normalise(path, rng):
    """One song as riddim's band: a list of (start_s, dur_s, pitch, instr)."""
    notes, progs, names, bpm = read(path)
    tempo = target_tempo(bpm)
    if tempo is None:
        return None, "tempo"
    beat = 60.0 / bpm
    rl = roles(notes, progs, names, beat)
    if LEAD not in rl.values() or DRUMS not in rl.values() or BASS not in rl.values():
        return None, "roles"
    stretch = bpm / tempo                              # new seconds = old * stretch
    out = []
    for ch, instr in rl.items():
        for s, d, p in notes[ch]:
            out.append((s * stretch, max(0.01, d * stretch), p, instr))
    if not out:
        return None, "empty"
    t0 = min(s for s, *_ in out)
    out = [(s - t0, d, p, i) for s, d, p, i in out]
    # one note per lead onset: the top one, so the lead is a line
    lead = {}
    for e in out:
        if e[3] == LEAD:
            k = round(e[0], 2)
            if k not in lead or e[2] > lead[k][2]:
                lead[k] = e
    out = [e for e in out if e[3] != LEAD] + list(lead.values())
    return out, tempo


def transpose(song, k):
    out = []
    for s, d, p, i in song:
        if i != DRUMS:
            p += k
            if not 0 <= p <= 127:
                continue
        out.append((s, d, p, i))
    return out


def to_events(song):
    ev = []
    for s, d, p, i in sorted(song):
        t = int(round(s * TIME_RESOLUTION))
        dt = max(1, min(MAX_DUR - 1, int(round(d * TIME_RESOLUTION))))
        ev += [TIME_OFFSET + t, DUR_OFFSET + dt, NOTE_OFFSET + i * 128 + p]
    return ev


# -- tokenizing ----------------------------------------------------------------

SEQ_EVENTS = EVENT_SIZE * M


def _window(tokens, z):
    seq = ops.translate(tokens, -ops.min_time(tokens, seconds=False), seconds=False)
    if ops.max_time(seq, seconds=False) >= MAX_TIME:
        return None
    return [z] + seq


def lead_windows(events, rng, count):
    """The inference task: context, then band as controls, lead as events."""
    end = ops.max_time(events, seconds=False)
    lead_lo = NOTE_OFFSET + LEAD * 128
    triples = list(zip([t - TIME_OFFSET for t in events[0::3]], events[1::3], events[2::3]))
    out = []
    lo, hi = 8 * TIME_RESOLUTION, end - 20 * TIME_RESOLUTION
    if hi <= lo:
        return out
    horizon = 60 * TIME_RESOLUTION          # more than a window's worth
    for _ in range(count):
        split = rng.randint(lo, hi)
        before = [x for x in triples if x[0] < split]
        ctx = before[-rng.randint(30, 150):]          # a prompt's worth of band and lead
        if not ctx:
            continue
        t0 = ctx[0][0]
        ev, ctl = [], []
        for t, d, n in ctx:
            ev += [TIME_OFFSET + t - t0, d, n]
        n_lead = 0
        for t, d, n in triples:
            if t < split or t >= split + horizon:
                continue
            if lead_lo <= n < lead_lo + 128:
                ev += [TIME_OFFSET + t - t0, d, n]
                n_lead += 1
            else:
                ctl += [CONTROL_OFFSET + TIME_OFFSET + t - t0, CONTROL_OFFSET + d, CONTROL_OFFSET + n]
        if n_lead < 8:
            continue
        ev = ops.pad(ev, split + horizon - t0)
        tokens, _ = ops.anticipate(ev, ctl)
        tokens = tokens[:SEQ_EVENTS]
        if len(tokens) < SEQ_EVENTS:
            continue
        w = _window(tokens, ANTICIPATE)
        if w:
            out.append(w)
    return out


def full_windows(events, rng, augment):
    """The model's own augmented tokenization of the whole normalised song."""
    instruments = list(ops.get_instruments(events).keys())
    end = ops.max_time(events, seconds=False)
    out = []
    for k in range(augment):
        np.random.seed(rng.randrange(2 ** 31))
        if k % 10 == 0:
            ev, ctl, z = events.copy(), [], AUTOREGRESS
        elif k % 10 == 1:
            (ev, ctl), z = extract_spans(events, 0.05), ANTICIPATE
        elif k % 10 < 6:
            (ev, ctl), z = extract_random(events, np.random.randint(1, 10)), ANTICIPATE
        else:
            u = 1 + np.random.randint(max(1, len(instruments) - 1))
            sub = np.random.choice(instruments, min(u, len(instruments)), replace=False)
            (ev, ctl), z = extract_instruments(events, sub), ANTICIPATE
        ev = ops.pad(ev, end)
        tokens, _ = ops.anticipate(ev, ctl)
        tokens = [SEPARATOR] * 3 + tokens
        for i in range(0, len(tokens) - SEQ_EVENTS + 1, SEQ_EVENTS):
            w = _window(tokens[i:i + SEQ_EVENTS], z)
            if w:
                out.append(w)
    return out


def signature(song):
    """The tune, not the file: Lakh holds several MIDI versions of many
    songs. The lead's first 40 intervals (transposition-free) name it."""
    lead = [p for _, _, p, i in sorted(song) if i == LEAD]
    iv = tuple(b - a for a, b in zip(lead, lead[1:]))[:40]
    return hashlib.md5(repr(iv).encode()).hexdigest() if len(iv) >= 20 else None


def work(args):
    path, variants, seed, full_aug = args
    rng = random.Random(seed)
    try:
        song, why = normalise(path, rng)
    except Exception as exc:
        return path, None, None, f"error:{type(exc).__name__}"
    if song is None:
        return path, None, None, why
    sig = signature(song)
    lead, full = [], []
    for v in range(variants):
        k = 0 if v == 0 else rng.randint(-6, 5)
        ev = to_events(transpose(song, k))
        if len(ev) < 3 * 100:
            continue
        lead += lead_windows(ev, rng, 3)
        if full_aug:
            full += full_windows(ev, rng, full_aug)
    pack = lambda ws: np.array([w for w in ws if len(w) == CONTEXT_SIZE], dtype=np.int32).reshape(-1, CONTEXT_SIZE)  # noqa: E731
    return path, pack(lead), pack(full), sig or "ok"


def main():
    scores_path, out_dir = sys.argv[1], sys.argv[2]

    def opt(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    min_score, max_score = float(opt("--min-score", 3.0)), float(opt("--max-score", 99))
    variants = int(opt("--variants", 3))
    full_aug = int(opt("--full-augment", 2))
    limit = int(opt("--limit", 0))
    jobs = int(opt("--jobs", 16))

    rows = [json.loads(line) for line in open(scores_path)]
    paths = sorted({r["path"] for r in rows if min_score <= r["score"] < max_score})
    if limit:
        paths = paths[:limit]
    print(f"{len(paths)} files with {min_score} <= score < {max_score}", flush=True)
    # valid: 4% of songs, chosen by hash so it is stable across rebuilds
    is_valid = lambda p: int(hashlib.md5(p.encode()).hexdigest(), 16) % 25 == 0  # noqa: E731
    for task in ("lead", "full"):
        os.makedirs(os.path.join(out_dir, task), exist_ok=True)
    pending = {(t, s): [] for t in ("lead", "full") for s in ("train", "valid")}
    counts = defaultdict(int)
    shards = defaultdict(int)

    def flush(task, split, force=False):
        arrs = pending[(task, split)]
        n = sum(len(a) for a in arrs)
        if not n or (not force and n < 20000):
            return
        arr = np.concatenate(arrs)
        np.random.default_rng(shards[task]).shuffle(arr)
        name = "valid.npy" if split == "valid" else f"train-{shards[task]:03d}.npy"
        if split == "valid":
            np.save(os.path.join(out_dir, task, name), arr)
        else:
            np.save(os.path.join(out_dir, task, name), arr)
            shards[task] += 1
        counts[f"{task}_{split}"] += len(arr)
        pending[(task, split)] = []

    reasons = defaultdict(int)
    seen = set()
    done = 0
    with ProcessPoolExecutor(jobs) as ex:
        for path, lead, full, why in ex.map(work, [(p, variants, i, full_aug) for i, p in enumerate(paths)],
                                            chunksize=8):
            done += 1
            if lead is not None and why != "ok":
                if why in seen:
                    reasons["duplicate"] += 1
                    continue
                seen.add(why)
                why = "ok"
            reasons[why] += 1
            if lead is not None:
                split = "valid" if is_valid(path) else "train"
                pending[("lead", split)].append(lead)
                pending[("full", split)].append(full)
                if split == "train":
                    flush("lead", "train")
                    flush("full", "train")
            if done % 2000 == 0:
                print(f"{done}/{len(paths)} {dict(reasons)} {dict(counts)}", flush=True)
    for key in pending:
        flush(*key, force=True)
    stats = {"files": len(paths), "reasons": dict(reasons), **counts}
    for k, v in list(counts.items()):
        stats[k + "_tokens_M"] = round(v * CONTEXT_SIZE / 1e6, 1)
    json.dump(stats, open(os.path.join(out_dir, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()

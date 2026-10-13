"""riddim's own songs as training data - ours to use, in exactly the
instruments, tempos and textures the model plays over at inference.

Only the "full" (general, augmented) tokenization: the model learns how a
riddim band moves - one-drops, steppers, half-time, 2-step, skanks, subs at
66-142 bpm - but is never taught to write riddim's leads, which is the part
it is there to improve on.

usage: riddim_corpus.py OUT_DIR [--songs 3000] [--jobs 16]
"""
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import build_data as B                        # noqa: E402
from anticipation.config import CONTEXT_SIZE  # noqa: E402

PROGRAMS = {"sub": B.BASS, "chords": B.CHORDS, "pad": B.PAD, "lead": B.LEAD}


def song_events(seed: int):
    from riddim.compose import compose
    from riddim.profile import PROFILES
    from riddim.render.midifile import GM_DRUMS
    from riddim.units import TICKS_PER_BEAT

    rng = random.Random(seed)
    profile = rng.choice(list(PROFILES))
    song = compose(profile=profile, seed=seed, live=rng.random() < 0.5)
    spt = 60.0 / song.bpm / TICKS_PER_BEAT
    out = []
    for part in song.parts:
        kit = part.output == "kit"
        instr = B.DRUMS if kit else PROGRAMS.get(part.role)
        if instr is None:
            continue
        for n in part.notes:
            p = GM_DRUMS.get(n.midi, n.midi) if kit else n.midi
            out.append((n.start * spt, n.length * spt, p, instr))
    return out


def work(seed):
    rng = random.Random(seed)
    try:
        song = song_events(seed)
    except Exception:
        return None
    ev = B.to_events(B.transpose(song, 0))
    ws = B.full_windows(ev, rng, 2)
    return np.array([w for w in ws if len(w) == CONTEXT_SIZE], dtype=np.int32).reshape(-1, CONTEXT_SIZE)


def main():
    out_dir = sys.argv[1]

    def opt(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    songs, jobs = int(opt("--songs", 3000)), int(opt("--jobs", 16))
    os.makedirs(os.path.join(out_dir, "full"), exist_ok=True)
    train, valid = [], []
    with ProcessPoolExecutor(jobs) as ex:
        for i, arr in enumerate(ex.map(work, range(100000, 100000 + songs), chunksize=4)):
            if arr is None or not len(arr):
                continue
            (valid if i % 25 == 0 else train).append(arr)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{songs} songs, {sum(len(a) for a in train)} sequences", flush=True)
    t = np.concatenate(train)
    np.random.default_rng(0).shuffle(t)
    for i, part in enumerate(np.array_split(t, 4)):
        np.save(os.path.join(out_dir, "full", f"train-{i:03d}.npy"), part)
    np.save(os.path.join(out_dir, "full", "valid.npy"), np.concatenate(valid))
    print(f"riddim corpus: {len(t)} train sequences ({len(t) * CONTEXT_SIZE / 1e6:.0f}M tokens)", flush=True)


if __name__ == "__main__":
    main()

"""Build the DMZ-leaning fine-tuning set: the Lakh files the content filter
scored >= THRESHOLD, converted and tokenized the way the Anticipatory Music
Transformer was trained (augmented, so it keeps its infilling/control mode).

usage: prep_finetune.py THRESHOLD OUT_DIR
"""
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

from anticipation.convert import midi_to_compound
from anticipation.tokenize import tokenize

ML_HOME = os.environ.get("ML_HOME", os.path.expanduser("~/ml"))

threshold, out_dir = float(sys.argv[1]), sys.argv[2]
os.makedirs(out_dir, exist_ok=True)


def convert(path):
    dest = os.path.join(out_dir, "compound", os.path.basename(path) + ".compound.txt")
    if os.path.exists(dest):
        return dest
    try:
        toks = midi_to_compound(path)
    except Exception:
        return None
    with open(dest, "w") as f:
        f.write(" ".join(str(t) for t in toks))
    return dest


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(os.path.join(ML_HOME, "out", "dmz_scores.jsonl"))]
    sel = [r["path"] for r in rows if r["score"] >= threshold]
    print(f"{len(sel)} files with score >= {threshold}", flush=True)
    os.makedirs(os.path.join(out_dir, "compound"), exist_ok=True)
    with ProcessPoolExecutor(16) as ex:
        done = [d for d in ex.map(convert, sel, chunksize=16) if d]
    print(f"{len(done)} converted", flush=True)
    random.Random(0).shuffle(done)
    n_valid = max(20, len(done) // 20)
    valid, train = done[:n_valid], done[n_valid:]
    # tokenize in 16 groups in parallel, like the reference script
    groups = [train[i::16] for i in range(16)]
    outs = [os.path.join(out_dir, f"train-{i:02d}.txt") for i in range(16)]
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(tokenize, groups, outs, [10] * 16, range(16)))
    tokenize(valid, os.path.join(out_dir, "valid.txt"), 1, 0)
    seqs = sum(r[0] for r in res)
    print(f"train sequences: {seqs} ({seqs * 1024 / 1e6:.1f}M tokens), valid files: {len(valid)}", flush=True)

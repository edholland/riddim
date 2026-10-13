"""Fine-tune the music transformer in MLX (~2x the PyTorch MPS loop here).

usage: train_mlx.py OUT_DIR --init MODEL_DIR --mix NAME=DIR:WEIGHT[,...] [options]

  --mix      training sets and how often to draw from each, e.g.
             lead=data/lead_v2:0.6,dmz=data/ft_dmz:0.3,riddim=data/riddim:0.1
             Each DIR holds train-*.npy (or train-*.txt) and valid.npy/.txt
             of 1024-token sequences.
  --steps    optimizer steps (default 3000)
  --batch    sequences per micro-batch (default 8)
  --accum    micro-batches per step (default 4)
  --lr       peak learning rate (default 2e-5), warmup 100, cosine to 10%
  --eval     evaluate every N steps (default 250); the first --mix set's
             validation loss picks `best`
  --start    resume: the step to carry on from (with --init OUT_DIR/latest);
             the schedule and data order continue from there
  --probe    20 steps, report speed and memory, save nothing

Writes OUT_DIR/latest, OUT_DIR/best (Hugging Face checkpoints that riddim's
worker loads as they are) and OUT_DIR/log.jsonl.
"""
import glob
import json
from functools import partial
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import mlx_gpt2  # noqa: E402

SEQ = 1024


def _opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def load_split(d: str, split: str) -> np.ndarray:
    """All sequences of a split as one int32 [N, 1024] array (cached as .npy)."""
    npys = sorted(glob.glob(os.path.join(d, f"{split}*.npy")))
    if npys:
        return np.concatenate([np.load(p, mmap_mode="r") for p in npys])
    arrays = []
    for p in sorted(glob.glob(os.path.join(d, f"{split}*.txt"))):
        rows = [r for r in (np.array(line.split(), dtype=np.int32) for line in open(p)) if len(r) == SEQ]
        if rows:
            a = np.stack(rows)
            np.save(p[:-4] + ".npy", a)
            arrays.append(a)
    if not arrays:
        raise SystemExit(f"no {split} data in {d}")
    return np.concatenate(arrays)


def main():
    out_dir = sys.argv[1]
    init = _opt("--init", None)
    mix_arg = _opt("--mix", None)
    if not init or not mix_arg:
        raise SystemExit(__doc__)
    steps = int(_opt("--steps", 3000))
    batch, accum = int(_opt("--batch", 8)), int(_opt("--accum", 4))
    lr, warmup = float(_opt("--lr", 2e-5)), 100
    every = int(_opt("--eval", 250))
    probe = "--probe" in sys.argv
    start = int(_opt("--start", 0))
    if probe:
        steps = 20

    mix = []
    for item in mix_arg.split(","):
        name, rest = item.split("=")
        d, w = rest.rsplit(":", 1)
        tr = load_split(d, "train")
        va = load_split(d, "valid")
        va = va[np.random.default_rng(1).permutation(len(va))[:128]]
        mix.append((name, tr, va, float(w)))
        print(f"{name}: {len(tr)} train, {len(va)} valid sequences, weight {w}", flush=True)
    weights = np.array([w for *_, w in mix])
    weights /= weights.sum()

    model = mlx_gpt2.load(init)
    model.compute_dtype = mx.bfloat16
    model.dropout = 0.1

    def schedule(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, steps - warmup))
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    opt = optim.AdamW(learning_rate=lr, betas=[0.9, 0.95], weight_decay=0.1)
    opt.init(model.trainable_parameters())      # compiled update needs its state to exist

    def loss_fn(m, x):
        logits = m(x[:, :-1]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, x[:, 1:], reduction="mean")

    grad_fn = nn.value_and_grad(model, loss_fn)

    # Compiled: ~1.4x the eager step (4.5k tok/s at batch 8 on an M5 Pro). The
    # whole step is one graph - accumulation, clipping and update inside it.
    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(xs):
        total, grads = 0.0, None
        for i in range(accum):
            loss, g = grad_fn(model, xs[i])
            grads = g if grads is None else _add(grads, g)
            total = total + loss
        if accum > 1:
            grads = _scale(grads, 1.0 / accum)
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return total / accum, norm

    def evaluate():
        model.eval()
        res = {}
        for name, _, va, _ in mix:
            ls = [loss_fn(model, mx.array(va[i:i + batch])).item() for i in range(0, len(va), batch)]
            res[name] = sum(ls) / len(ls)
        model.train()
        return res

    rng = np.random.default_rng(start)
    cursors = [0] * len(mix)
    perms = [rng.permutation(len(tr)) for _, tr, _, _ in mix]

    def next_batch():
        rows = []
        for _ in range(batch):
            k = rng.choice(len(mix), p=weights)
            tr = mix[k][1]
            if cursors[k] >= len(tr):
                perms[k], cursors[k] = rng.permutation(len(tr)), 0
            rows.append(np.asarray(tr[perms[k][cursors[k]]]))
            cursors[k] += 1
        return mx.array(np.stack(rows))

    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    best = float("inf")
    best_file = os.path.join(out_dir, "best.json")
    if start and os.path.exists(best_file):
        best = json.load(open(best_file))["loss"]
    if not probe and not start:
        v = evaluate()
        best = v[mix[0][0]]
        print(f"step 0 valid {json.dumps({k: round(x, 4) for k, x in v.items()})}", flush=True)
        log.write(json.dumps({"step": 0, "valid": v}) + "\n")
    model.train()
    t0 = time.time()
    for step in range(start, steps):
        opt.learning_rate = mx.array(schedule(step))
        xs = mx.stack([next_batch() for _ in range(accum)])
        loss, norm = train_step(xs)
        mx.eval(state, loss)
        total = loss.item()
        if probe or step % 20 == 0:
            done = (step - start + 1) * batch * accum * SEQ
            rate = done / (time.time() - t0)
            mem = mx.get_peak_memory() / 1e9
            eta = (steps - step - 1) * batch * accum * SEQ / rate / 3600
            print(f"step {step} loss {total:.4f} lr {schedule(step):.2e} gnorm {norm.item():.2f} "
                  f"{rate:.0f} tok/s peak {mem:.1f}GB eta {eta:.1f}h", flush=True)
            log.write(json.dumps({"step": step, "loss": total, "tok_s": rate}) + "\n")
            log.flush()
        if not probe and step and (step % every == 0 or step == steps - 1):
            v = evaluate()
            print(f"step {step} valid {json.dumps({k: round(x, 4) for k, x in v.items()})}", flush=True)
            log.write(json.dumps({"step": step, "valid": v}) + "\n")
            log.flush()
            mlx_gpt2.save(model, os.path.join(out_dir, "latest"))
            json.dump({"step": step}, open(os.path.join(out_dir, "latest", "step.json"), "w"))
            if v[mix[0][0]] < best:
                best = v[mix[0][0]]
                mlx_gpt2.save(model, os.path.join(out_dir, "best"))
                json.dump({"step": step, "loss": best, "valid": v}, open(best_file, "w"))
                print(f"  best so far ({mix[0][0]} {best:.4f})", flush=True)
    if not probe:
        print("DONE", flush=True)


def _add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def _scale(a, s):
    from mlx.utils import tree_map
    return tree_map(lambda x: x * s, a)


if __name__ == "__main__":
    main()

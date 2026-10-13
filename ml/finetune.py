"""Fine-tune the Anticipatory Music Transformer on the DMZ-leaning set (MPS).

usage: finetune.py DATA_DIR OUT_DIR [MAX_STEPS] [--probe]
  --probe: 10 steps, report tokens/sec and memory, save nothing.
"""
import glob
import math
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM

ML_HOME = os.environ.get("ML_HOME", os.path.expanduser("~/ml"))

BASE = os.path.join(ML_HOME, "models", "music-medium-800k")
data_dir, out_dir = sys.argv[1], sys.argv[2]
max_steps = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1600
probe = "--probe" in sys.argv
BATCH, ACCUM, LR, WARMUP = 2, 16, 2e-5, 100
dev = torch.device("mps")


def load(paths, limit=None, seed=0):
    seqs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                toks = [int(t) for t in line.split()]
                if len(toks) == 1024:
                    seqs.append(toks)
    random.Random(seed).shuffle(seqs)
    return seqs[:limit] if limit else seqs


def main():
    need = max_steps * BATCH * ACCUM
    train = load(sorted(glob.glob(os.path.join(data_dir, "train-*.txt"))), need)
    valid = load([os.path.join(data_dir, "valid.txt")], 64, seed=1)
    print(f"train sequences {len(train)} (need {need}), valid {len(valid)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE, attn_implementation="sdpa").to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

    def lr_at(step):
        if step < WARMUP:
            return LR * (step + 1) / WARMUP
        p = (step - WARMUP) / max(1, max_steps - WARMUP)
        return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for i in range(0, len(valid), BATCH):
            x = torch.tensor(valid[i:i + BATCH], device=dev)
            with torch.autocast("mps", dtype=torch.bfloat16):
                losses.append(model(input_ids=x, labels=x).loss.float().item())
        model.train()
        return sum(losses) / len(losses)

    if not probe:
        os.makedirs(out_dir, exist_ok=True)
        print(f"step 0 valid loss {evaluate():.4f}", flush=True)
    t0 = time.time()
    cursor = 0
    steps = 10 if probe else max_steps
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        total = 0.0
        for _ in range(ACCUM):
            x = torch.tensor(train[cursor % len(train): cursor % len(train) + BATCH], device=dev)
            cursor += BATCH
            with torch.autocast("mps", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss / ACCUM
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if probe or step % 20 == 0:
            done = (step + 1) * BATCH * ACCUM * 1024
            rate = done / (time.time() - t0)
            mem = torch.mps.driver_allocated_memory() / 1e9
            eta = (steps - step - 1) * BATCH * ACCUM * 1024 / rate / 3600
            print(f"step {step} loss {total:.4f} lr {lr_at(step):.2e} {rate:.0f} tok/s mem {mem:.1f}GB "
                  f"eta {eta:.1f}h", flush=True)
        if not probe and step and step % 200 == 0:
            print(f"step {step} valid loss {evaluate():.4f}", flush=True)
            model.save_pretrained(os.path.join(out_dir, "latest"))
    if not probe:
        print(f"final valid loss {evaluate():.4f}", flush=True)
        model.save_pretrained(os.path.join(out_dir, "final"))
        print("DONE", flush=True)


if __name__ == "__main__":
    main()

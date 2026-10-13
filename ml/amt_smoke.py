"""Smoke test: load the base model and generate 20 s with the reference
sampler (slow - the baseline fast_smoke.py is measured against)."""
import os
import time

import torch
from anticipation.convert import events_to_midi
from anticipation.sample import generate
from transformers import AutoModelForCausalLM

ML_HOME = os.environ.get("ML_HOME", os.path.expanduser("~/ml"))
dev = "mps"
t = time.time()
model = AutoModelForCausalLM.from_pretrained(os.path.join(ML_HOME, "models", "music-medium-800k")).to(dev).eval()
print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.0f}M params in {time.time()-t:.1f}s on {dev}")
torch.manual_seed(1)
t = time.time()
events = generate(model, start_time=0, end_time=20, top_p=0.98)
dt = time.time() - t
print(f"generated 20s of music: {len(events)//3} events ({len(events)} tokens) in {dt:.1f}s = {len(events)/dt:.0f} tok/s")
events_to_midi(events).save(os.path.join(ML_HOME, "out", "amt_smoke.mid"))
print("saved out/amt_smoke.mid")

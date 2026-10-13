"""Smoke test: the KV-cached sampler on the base model, 20 s and 60 s."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from anticipation.convert import events_to_midi     # noqa: E402
from transformers import AutoModelForCausalLM       # noqa: E402

from fast_sample import generate_fast               # noqa: E402

ML_HOME = os.environ.get("ML_HOME", os.path.expanduser("~/ml"))
model = AutoModelForCausalLM.from_pretrained(os.path.join(ML_HOME, "models", "music-medium-800k")).to("mps").eval()
for secs in (20, 60):
    torch.manual_seed(1)
    t = time.time()
    events = generate_fast(model, 0, secs, top_p=0.98)
    dt = time.time() - t
    print(f"{secs}s of music: {len(events)//3} events, {len(events)} tokens in {dt:.1f}s = {len(events)/dt:.0f} tok/s")
    events_to_midi(events).save(os.path.join(ML_HOME, "out", f"fast_{secs}s.mid"))

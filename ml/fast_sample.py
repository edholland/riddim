"""A KV-cached sampler for the Anticipatory Music Transformer.

The reference `anticipation.sample.generate` runs three full forward passes
over up to 1017 tokens of history for every event it writes (and re-times
the history window as it slides, which is why it never caches). This keeps
the model's key/value cache across events and only rebuilds it when the
window has to slide - by a chunk, not one event at a time.

Same constraints as the reference: token-slot masking, no events in the
past, at most 16 instruments, nucleus sampling, anticipated controls
interleaved `delta` ahead.
"""
import math

import torch
import torch.nn.functional as F

from anticipation import ops
from anticipation.config import DELTA, TIME_RESOLUTION
from anticipation.sample import future_logits, instr_logits, nucleus, safe_logits
from anticipation.vocab import ANTICIPATE, ATIME_OFFSET, AUTOREGRESS, CONTROL_OFFSET, TIME_OFFSET

WINDOW = 1017        # tokens of history the model sees (as in the reference)
KEEP = 765           # after a slide, keep this much (a multiple of 3)


class Sampler:
    def __init__(self, model, z):
        self.model, self.z = model, z
        self.device = model.device
        self.reset_cache()

    def reset_cache(self):
        self.past = None
        self.fed = 0             # tokens of `tokens` already in the cache
        self.ws = 0              # window start in `tokens`
        self.offset = 0
        self.logits = None

    def _rel(self, tokens, i):
        t = tokens[i]
        return t - self.offset if (i - self.ws) % 3 == 0 else t

    @torch.no_grad()
    def _feed(self, ids):
        if not ids:
            return
        x = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=x, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        self.logits = out.logits[0, -1].float()

    def sync(self, tokens):
        """Bring the cache up to date with `tokens` (sliding if needed)."""
        if len(tokens) - self.ws > WINDOW or self.past is None:
            if len(tokens) - self.ws > WINDOW:
                self.ws = max(0, len(tokens) - KEEP)
                self.ws -= self.ws % 3
            self.past, self.fed, self.logits = None, self.ws, None
            self.offset = ops.min_time(tokens[self.ws:], seconds=False) if len(tokens) > self.ws else 0
            self._feed(list(self.z) + [self._rel(tokens, i) for i in range(self.ws, len(tokens))])
            self.fed = len(tokens)
            return
        self._feed([self._rel(tokens, i) for i in range(self.fed, len(tokens))])
        self.fed = len(tokens)

    @torch.no_grad()
    def add_event(self, tokens, top_p, current_time):
        self.sync(tokens)
        new = []
        for i in range(3):
            logits = self.logits.clone()
            idx = len(self.z) + (len(tokens) - self.ws) + i - 1
            logits = safe_logits(logits, idx)
            if i == 0:
                logits = future_logits(logits, current_time - self.offset)
            elif i == 2:
                logits = instr_logits(logits, tokens)
            logits = nucleus(logits, top_p)
            tok = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
            new.append(tok)
            self._feed([tok])
        self.fed += 3
        new[0] += self.offset
        return new


def generate_fast(model, start_time, end_time, inputs=None, controls=None, top_p=0.98,
                  delta=DELTA * TIME_RESOLUTION):
    inputs = inputs or []
    controls = controls or []
    start_time = int(TIME_RESOLUTION * start_time)
    end_time = int(TIME_RESOLUTION * end_time)

    prompt = ops.pad(ops.clip(inputs, 0, start_time, clip_duration=False, seconds=False), start_time)
    future = ops.clip(inputs, start_time + 1, ops.max_time(inputs, seconds=False), clip_duration=False,
                      seconds=False)
    controls = ops.clip(controls, DELTA, ops.max_time(controls, seconds=False), clip_duration=False,
                        seconds=False)
    z = [ANTICIPATE] if controls or future else [AUTOREGRESS]
    tokens, controls = ops.anticipate(prompt, ops.sort(controls + [CONTROL_OFFSET + t for t in future]))
    current_time = ops.max_time(prompt, seconds=False)

    sampler = Sampler(model, z)
    if controls:
        atime, adur, anote = controls[0:3]
        anticipated = controls[3:]
        anticipated_time = atime - ATIME_OFFSET
    else:
        anticipated_time = math.inf

    while True:
        while current_time >= anticipated_time - delta:
            tokens.extend([atime, adur, anote])
            if anticipated:
                atime, adur, anote = anticipated[0:3]
                anticipated = anticipated[3:]
                anticipated_time = atime - ATIME_OFFSET
            else:
                anticipated_time = math.inf
        sampler.fed = min(sampler.fed, len(tokens))
        new = sampler.add_event(tokens, top_p, max(start_time, current_time))
        new_time = new[0] - TIME_OFFSET
        if new_time >= end_time:
            break
        tokens.extend(new)
        current_time = new_time

    events, _ = ops.split(tokens)
    return ops.sort(ops.unpad(events) + future)

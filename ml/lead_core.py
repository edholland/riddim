"""The constrained lead sampler shared by the A/B script and the worker.

The Anticipatory Music Transformer writes one instrument (a saw lead,
GM 81) over a band given as anticipated controls. It may only play
inside a register, only in the song's pitch classes, and never start a
note closer than `min_gap` (10 ms units) to the last one - the model's
raw leads are dense and chromatic, these three keep it a hook.
"""
import torch

from anticipation.config import MAX_TIME, TIME_RESOLUTION
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET

import fast_sample

LEAD_INSTR = 81          # GM sawtooth lead


class LeadOnly(fast_sample.Sampler):
    """Only the lead instrument, only inside its register."""

    def __init__(self, model, z, lo, hi, scale_pcs=None, min_gap=0, grid=None, max_repeat=0,
                 rhythm=None, stop=None):
        super().__init__(model, z)
        # rhythm = [(onset, duration), ...] in 10ms units: the times are given,
        # the model only chooses the pitches; after the last, time jumps to `stop`
        self.rhythm, self.stop, self.k = rhythm, stop, 0
        self.max_repeat = max_repeat    # no more than this many of one pitch in a row
        self.pitches = []
        # grid = (origin, step) in 10ms units: onsets only on it, so the model
        # hears its own rhythm as it will be played rather than being snapped after
        self.on_grid = None
        if grid is not None:
            origin, step = grid
            on = torch.zeros(2 * MAX_TIME, dtype=torch.bool)
            k = 0
            while origin + k * step < 2 * MAX_TIME:
                t = int(round(origin + k * step))
                # +-10ms: the band's own onsets were rounded on their own
                on[max(0, t - 1):t + 2] = True
                k += 1
            self.on_grid = on.to(self.device)
        self.allowed = torch.zeros(self.model.config.vocab_size, dtype=torch.bool)
        for p in range(lo, hi + 1):
            if scale_pcs is None or p % 12 in scale_pcs:      # in the song's mode only
                self.allowed[NOTE_OFFSET + LEAD_INSTR * 128 + p] = True
        self.allowed = self.allowed.to(self.device)
        self.min_gap = min_gap          # 10ms units between lead onsets
        self.last = None                # absolute time of the last lead onset

    @torch.no_grad()
    def add_event(self, tokens, top_p, current_time):
        self.sync(tokens)
        new = []
        for i in range(3):
            logits = self.logits.clone()
            idx = len(self.z) + (len(tokens) - self.ws) + i - 1
            logits = fast_sample.safe_logits(logits, idx)
            if i == 0 and self.rhythm is not None:
                t = self.rhythm[self.k][0] if self.k < len(self.rhythm) else self.stop
                keep = logits[TIME_OFFSET + t - self.offset].clone()
                logits[:] = -float("inf")
                logits[TIME_OFFSET + t - self.offset] = keep if torch.isfinite(keep) else 0.0
            elif i == 1 and self.rhythm is not None and self.k < len(self.rhythm):
                d = DUR_OFFSET + self.rhythm[self.k][1]
                logits[:] = -float("inf")
                logits[d] = 0.0
            elif i == 0:
                logits = fast_sample.future_logits(logits, current_time - self.offset)
                if self.last is not None and self.min_gap:
                    # room between notes: no onset closer than min_gap to the last
                    lo_t = max(0, self.last - self.offset)
                    logits[TIME_OFFSET + lo_t:TIME_OFFSET + lo_t + self.min_gap] = -float("inf")
                if self.on_grid is not None:
                    off = ~self.on_grid[self.offset:self.offset + MAX_TIME]
                    logits[TIME_OFFSET:TIME_OFFSET + MAX_TIME][off] = -float("inf")
            elif i == 2:
                logits[~self.allowed] = -float("inf")
                run = self.pitches[-self.max_repeat:] if self.max_repeat else []
                if run and len(run) == self.max_repeat and len(set(run)) == 1 \
                        and self.allowed.sum() > 1:
                    logits[run[0]] = -float("inf")
            logits = fast_sample.nucleus(logits, top_p)
            tok = int(torch.multinomial(torch.softmax(logits, -1), 1))
            new.append(tok)
            self._feed([tok])
        self.fed += 3
        new[0] += self.offset
        self.last = new[0] - TIME_OFFSET
        self.pitches.append(new[2])
        self.k += 1
        return new


def write_lead(model, controls, start_s, end_s, lo, hi, top_p, seed, scale_pcs=None, min_gap=0,
               grid=None, max_repeat=0, rhythm=None):
    torch.manual_seed(seed)
    orig = fast_sample.Sampler
    fast_sample.Sampler = lambda m, z: LeadOnly(
        m, z, lo, hi, scale_pcs, min_gap, grid, max_repeat, rhythm,
        int(round(end_s * TIME_RESOLUTION)) + 1)
    try:
        events = fast_sample.generate_fast(model, start_s, end_s, inputs=controls, top_p=top_p)
    finally:
        fast_sample.Sampler = orig
    out = []
    for t, d, n in zip(events[0::3], events[1::3], events[2::3]):
        note = n - NOTE_OFFSET
        if note // 128 == LEAD_INSTR:
            out.append(((t - TIME_OFFSET) / TIME_RESOLUTION, (d - DUR_OFFSET) / TIME_RESOLUTION, note % 128))
    return out

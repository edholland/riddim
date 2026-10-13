"""The model-lead worker: keeps the Anticipatory Music Transformer loaded
and writes leads for riddim, one JSON request per line on stdin, one JSON
reply per line on stdout (anything else goes to stderr).

Run by riddim.model_lead with the ML environment's Python, so riddim
itself never needs torch:

    ~/ml/venv/bin/python ml/lead_worker.py MODEL_DIR

Requests:
    {"op": "ping"}
        -> {"ok": true, "model": MODEL_DIR}
    {"op": "lead",
     "events": [[instr, pitch, start_s, dur_s], ...],   # the band, and any
                                                        # lead (instr 81) before `start`
     "start": s, "end": s,           # write the lead in [start, end)
     "lo": midi, "hi": midi,         # its register
     "scale": [pc, ...],             # pitch classes it may use
     "min_gap": s,                   # room between onsets
     "grid": [origin_s, step_s],     # optional: onsets only on this grid
     "max_repeat": n,                # optional: at most n of one pitch in a row
     "rhythm": [[start_s, dur_s], ...],  # optional: the onsets are given, the
                                         # model chooses only the pitches
     "top_p": 0.9, "seed": n}
        -> {"notes": [[start_s, dur_s, pitch], ...], "seconds": t}
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import torch                                            # noqa: E402
from anticipation import ops                            # noqa: E402
from anticipation.config import TIME_RESOLUTION         # noqa: E402
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET  # noqa: E402
from transformers import AutoModelForCausalLM           # noqa: E402

from lead_core import write_lead                        # noqa: E402


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def tokens(events):
    out = []
    for instr, pitch, start, dur in events:
        t = max(0, int(round(start * TIME_RESOLUTION)))
        d = max(1, min(999, int(round(dur * TIME_RESOLUTION))))
        out += [TIME_OFFSET + t, DUR_OFFSET + d, NOTE_OFFSET + int(instr) * 128 + int(pitch)]
    return ops.sort(out)


def lead(model, req):
    t = time.time()
    notes = write_lead(
        model, tokens(req["events"]), float(req["start"]), float(req["end"]),
        int(req["lo"]), int(req["hi"]), float(req.get("top_p", 0.9)), int(req.get("seed", 0)),
        set(req["scale"]) if req.get("scale") else None,
        int(round(float(req.get("min_gap", 0)) * TIME_RESOLUTION)),
        [g * TIME_RESOLUTION for g in req["grid"]] if req.get("grid") else None,
        int(req.get("max_repeat", 0)),
        [(int(round(s * TIME_RESOLUTION)), max(1, min(999, int(round(d * TIME_RESOLUTION)))))
         for s, d in req["rhythm"]] if req.get("rhythm") is not None else None,
    )
    return {"notes": [[s, d, p] for s, d, p in notes], "seconds": round(time.time() - t, 2)}


def main():
    model_dir = sys.argv[1]
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device()).eval()
    out = sys.stdout
    sys.stdout = sys.stderr          # stray prints must not corrupt the protocol
    out.write(json.dumps({"ready": True, "model": model_dir}) + "\n")
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "ping":
                reply = {"ok": True, "model": model_dir}
            elif req.get("op") == "lead":
                reply = lead(model, req)
            else:
                reply = {"error": f"unknown op {req.get('op')!r}"}
        except Exception as exc:     # report, keep serving
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()

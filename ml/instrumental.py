"""Which reference records are instrumental: for judging riddim, which has no
voice yet, against records without one.

Most roots references carry a singer or an MC (Demucs vocals above -15 dB on
58% of roots_dub and 70% of roots_reggae, against 34-38% of steppers and
digital dub), so an instrumental riddim sits nearer the mostly instrumental
families' centroids whatever its band sounds like (docs/roots-notes.md).

A record counts as instrumental by its Demucs vocal stem when it has one
(references/stats/stems.jsonl: under -25 dB against the mix), and otherwise
by CLAP (ml/judge.py's embeddings): it sits nearer "an instrumental track"
than "a singer" or "an MC" by more than THRESHOLD. On the 459 records with
both, the CLAP score ranks with the stem level at 0.71, and of the records it
calls instrumental, 71% are under -25 dB and 10% have a clear vocal (over
-15 dB).

    python ml/instrumental.py          # how many records a family keeps
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]

REFS = os.path.join(REPO, "references")
VOCAL = ("a song with a singer singing vocals", "a reggae vocal with a lead singer",
         "an MC toasting and chatting on the microphone")
INSTRUMENTAL = ("an instrumental track with no vocals", "an instrumental dub version with no singing")
THRESHOLD = -0.066     # the CLAP vocal-minus-instrumental score below which a record is instrumental
STEM_DB = -25.0        # Demucs vocals below this against the mix: instrumental


def _key(refs: str, file: str) -> str:
    path = os.path.abspath(os.path.join(refs, file))
    return f"{path}::{os.path.getsize(path)}"


def vocal_score(emb: np.ndarray, text: dict) -> float:
    import judge
    e = judge._unit(emb)
    return float((e @ text["voc"].T).max(1).mean() - (e @ text["ins"].T).max(1).mean())


def instrumental(refs: str = REFS) -> dict[str, set[str]]:
    """Per family, the judge-embedding keys (path::size) of its instrumental
    records."""
    import judge
    text = {"voc": judge._unit(judge.embed_text(list(VOCAL))),
            "ins": judge._unit(judge.embed_text(list(INSTRUMENTAL)))}
    stats = os.path.join(refs, "stats")
    plan = {p["id"]: p for p in (json.loads(line) for line in open(os.path.join(stats, "plan.jsonl")))}
    stem_db = {}
    path = os.path.join(stats, "stems.jsonl")
    if os.path.exists(path):
        for r in (json.loads(line) for line in open(path)):
            if "error" not in r and r["id"] in plan and os.path.exists(os.path.join(refs, plan[r["id"]]["file"])):
                stem_db[_key(refs, plan[r["id"]]["file"])] = r["vocals"]
    out = {}
    jdir = os.path.join(refs, "judge")
    for fam, path in judge.ref_files(jdir).items():
        keep = set()
        for k, v in np.load(path, allow_pickle=True).items():
            if not len(v):
                continue
            if (stem_db[k] < STEM_DB) if k in stem_db else (vocal_score(v, text) < THRESHOLD):
                keep.add(k)
        out[fam] = keep
    return out


def main() -> int:
    keep = instrumental()
    for fam, ks in keep.items():
        import judge
        path = judge.ref_files(os.path.join(REFS, "judge"))[fam]
        total = sum(1 for v in np.load(path, allow_pickle=True).values() if len(v))
        print(f"{fam:13s} {len(ks):4d} of {total} instrumental")
    return 0


if __name__ == "__main__":
    sys.exit(main())

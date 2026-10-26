"""The judge: how much does a recording sound like the records we love?

Ears for a model that has none. Every recording is cut into 10-second
windows and embedded by CLAP (laion/clap-htsat-unfused, a contrastive
audio-text model trained on hundreds of thousands of real recordings), then:

- **FAD** - Frechet Audio Distance between a set of renders and the
  reference set (the standard score for generated music: how far the two
  clouds of embeddings sit apart; lower is closer, 0 is indistinguishable
  to the model);
- **closeness** - each track's mean cosine similarity to the reference
  centroid, to rank single songs;
- **real-vs-machine** - zero-shot: the probability CLAP gives to
  descriptions of a produced record over descriptions of a computer /
  MIDI rendering. A crude detector, but a direct one.

    python ml/judge.py embed references/audio/*.mp3 --out refs.npz
    python ml/judge.py score refs.npz renders/*.wav

Embeddings are cached per file (by path and size) next to the --out file, so
re-scoring a new batch of renders only embeds the renders.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

MODEL = "laion/clap-htsat-unfused"
SR = 48000
WIN = 10.0

REAL = ["a professionally produced dubstep record, mixed and mastered",
        "a heavy sound system dub record with deep sub bass",
        "an underground UK bass music track, recorded in a studio"]
MACHINE = ["a cheap MIDI file played back with default synthesizer sounds",
           "computer generated music, a robotic loop with sine waves",
           "a video game soundtrack made of simple synthesized beeps"]

_model = None


def _load_model():
    global _model
    if _model is None:
        import torch
        from transformers import ClapModel, ClapProcessor
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        _model = (ClapModel.from_pretrained(MODEL).eval(), ClapProcessor.from_pretrained(MODEL))
    return _model


def decode(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
                          "-ar", str(sr), "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def windows(x, sr=SR, win=WIN, skip=0.0):
    """10 s windows over the body of the track (the first and last 5% - the
    DJ intro and the fade - are left out unless the track is short)."""
    n = int(win * sr)
    edge = int(len(x) * 0.05) if len(x) > 6 * n else 0
    x = x[edge:len(x) - edge] if edge else x
    return [x[i:i + n] for i in range(0, len(x) - n + 1, n)]


def embed(path, batch=8):
    import torch
    model, proc = _load_model()
    ws = windows(decode(path))
    out = []
    for i in range(0, len(ws), batch):
        inp = proc(audio=ws[i:i + batch], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            e = model.get_audio_features(**inp)
        e = getattr(e, "pooler_output", e)
        out.append(e.numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)


def embed_text(texts):
    import torch
    model, proc = _load_model()
    inp = proc(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        e = model.get_text_features(**inp)
    e = getattr(e, "pooler_output", e)
    return e.numpy()


def _cache_key(path):
    return f"{os.path.abspath(path)}::{os.path.getsize(path)}"


def embed_many(paths, cache_path):
    cache = dict(np.load(cache_path, allow_pickle=True)) if os.path.exists(cache_path) else {}
    out = {}
    for p in paths:
        k = _cache_key(p)
        if k not in cache:
            try:
                cache[k] = embed(p)
            except subprocess.CalledProcessError as e:  # a corrupt file never stops the run
                print(f"  skipping {p}: ffmpeg exit {e.returncode}", flush=True)
                cache[k] = np.zeros((0, 512), dtype=np.float32)
            np.savez(cache_path, **cache)
        out[p] = cache[k]
    return out


def fad(a, b):
    """Frechet distance between two sets of embeddings (rows)."""
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.real(linalg.sqrtm(s1.dot(s2)))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def real_prob(emb):
    """Mean probability, over windows, that CLAP picks a 'real record'
    description over a 'computer music' one."""
    t = _unit(embed_text(REAL + MACHINE))
    logits = 100.0 * _unit(emb) @ t.T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return float(p[:, :len(REAL)].sum(1).mean())


def score(ref_emb: dict, test_emb: dict) -> dict:
    ref = np.concatenate(list(ref_emb.values()))
    centre = _unit(ref.mean(0))
    rows = {}
    for p, e in test_emb.items():
        if len(e) == 0:
            continue
        rows[p] = {"closeness": float((_unit(e) @ centre).mean()), "real": real_prob(e),
                   "windows": len(e)}
    test = np.concatenate([e for e in test_emb.values() if len(e)])
    return {"fad": fad(ref, test), "tracks": rows}


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("embed", "score"):
        print(__doc__)
        return 1
    if a[0] == "embed":
        out = a[a.index("--out") + 1]
        paths = [p for p in a[1:] if p != "--out" and p != out]
        embed_many(paths, out)
        print(f"embedded {len(paths)} files into {out}")
        return 0
    refs = dict(np.load(a[1], allow_pickle=True))
    paths = a[2:]
    test = embed_many(paths, a[1].replace(".npz", "_tests.npz"))
    res = score(refs, test)
    ref_self = score(refs, {k: v for k, v in list(refs.items())})
    print(f"FAD vs references: {res['fad']:.3f}")
    for p, r in sorted(res["tracks"].items(), key=lambda kv: -kv[1]["closeness"]):
        print(f"  {r['closeness']:.3f} close  {r['real']:.2f} real   {os.path.basename(p)}")
    print(json.dumps({"fad": res["fad"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

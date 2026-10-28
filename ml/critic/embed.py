"""Audio embeddings for the critic: several backbones over the same windows.

Every file is decoded once, loudness-normalised (judge.NORM_DB over the
whole track), and cut into up to WINDOWS evenly spaced 10 s windows over its
body (the first and last 5% left out, as ml/judge.py does). Each backbone
embeds those same windows:

    clap        laion/clap-htsat-unfused   - the judge's CLAP (general audio)
    clap_music  laion/larger_clap_music    - LAION's CLAP trained on music
    ast         MIT/ast-finetuned-audioset - AudioSet's spectrogram transformer
                                             (mean of the last hidden layer)

(MERT is not here: its checkpoint needs trust_remote_code.)

Caches: <cache_dir>/<backbone>/<name>.npz, keyed by path and size like
judge.embed_many, so a re-run embeds only what is new.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
sys.path[:0] = [os.path.dirname(ML), ML]

import judge  # noqa: E402

SR = 48000
WIN = 10.0
WINDOWS = 12
BACKBONES = ("clap", "clap_music", "ast")
_MODELS: dict = {}


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _model(name: str):
    if name not in _MODELS:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        if name in ("clap", "clap_music"):
            from transformers import ClapModel, ClapProcessor
            ckpt = judge.MODEL if name == "clap" else "laion/larger_clap_music"
            _MODELS[name] = (ClapModel.from_pretrained(ckpt).eval().to(_device()), ClapProcessor.from_pretrained(ckpt))
        elif name == "ast":
            from transformers import ASTFeatureExtractor, ASTModel
            ckpt = "MIT/ast-finetuned-audioset-10-10-0.4593"
            _MODELS[name] = (ASTModel.from_pretrained(ckpt).eval().to(_device()), ASTFeatureExtractor.from_pretrained(ckpt))
        else:
            raise ValueError(name)
    return _MODELS[name]


def windows(path: str, n: int = WINDOWS) -> list[np.ndarray]:
    """Up to `n` evenly spaced 10 s windows of the loudness-normalised track."""
    x = judge.decode(path, SR, norm=True)
    w = int(WIN * SR)
    edge = int(len(x) * 0.05) if len(x) > 6 * w else 0
    x = x[edge:len(x) - edge] if edge else x
    if len(x) < w:
        return []
    starts = np.linspace(0, len(x) - w, min(n, max(1, (len(x) - w) // w + 1))).astype(int)
    return [x[s:s + w] for s in starts]


def embed_windows(name: str, ws: list[np.ndarray], batch: int = 8) -> np.ndarray:
    import torch
    if not ws:
        return np.zeros((0, 1), dtype=np.float32)
    model, proc = _model(name)
    dev = _device()
    out = []
    for i in range(0, len(ws), batch):
        chunk = ws[i:i + batch]
        with torch.no_grad():
            if name.startswith("clap"):
                inp = proc(audio=chunk, sampling_rate=SR, return_tensors="pt").to(dev)
                e = model.get_audio_features(**inp)
                e = getattr(e, "pooler_output", e)
            else:
                from scipy.signal import resample_poly
                chunk = [resample_poly(c, 1, 3).astype(np.float32) for c in chunk]   # 48 -> 16 kHz
                inp = proc(chunk, sampling_rate=16000, return_tensors="pt").to(dev)
                e = model(**inp).last_hidden_state.mean(1)
        out.append(e.float().cpu().numpy())
    return np.concatenate(out)


def embed_files(paths: list[str], cache_dir: str, name: str, backbones=BACKBONES) -> dict[str, dict]:
    """{backbone: {path: (windows, dim)}} for `paths`, through the caches
    <cache_dir>/<backbone>/<name>.npz."""
    caches, files = {}, {}
    for b in backbones:
        os.makedirs(os.path.join(cache_dir, b), exist_ok=True)
        files[b] = os.path.join(cache_dir, b, name + ".npz")
        caches[b] = dict(np.load(files[b], allow_pickle=True)) if os.path.exists(files[b]) else {}
    out = {b: {} for b in backbones}
    dirty = set()
    for i, p in enumerate(paths):
        key = judge._cache_key(p)
        need = [b for b in backbones if key not in caches[b]]
        if need:
            try:
                ws = windows(p)
            except subprocess.CalledProcessError:
                ws = []
            for b in need:
                caches[b][key] = embed_windows(b, ws)
                dirty.add(b)
        for b in backbones:
            out[b][p] = caches[b][key]
        if dirty and (i + 1) % 25 == 0:
            for b in dirty:
                np.savez(files[b], **caches[b])
            dirty.clear()
            print(f"    {i + 1}/{len(paths)}", flush=True)
    for b in dirty:
        np.savez(files[b], **caches[b])
    return out

"""The critic's training set: riddim's songs (with near-misses) and the
records, embedded the way ml/critic/embed.py does, mix and stems.

    python ml/critic/dataset.py ours --per-profile 300     # render + embed riddim's songs
    python ml/critic/dataset.py stems                      # embed the records' stems
    python ml/critic/dataset.py demucs --per-profile 100   # ours split by Demucs, as the records were

**Ours.** (Each mix goes through a record's codec first - lossy() - before
it is embedded or split.) For each profile, songs from seed 1000. Each is rendered with every
knob we have tuned as the branch has it, except that each knob is put back
to its old setting with probability OLD (0.2), independently. So the critic
sees today's songs and their near-misses, and cannot learn "riddim = this one
setting". The mix and riddim's own stems (render(stems_out=...): drums, bass,
other) are written to a temp folder, embedded and deleted. Only the
embeddings are kept, in references/critic/train/<backbone>/<profile>.npz
(keys "<seed>/mix", "<seed>/drums", ...), with each song's knobs in
references/critic/train/songs.jsonl.

**Records' stems.** The drums, bass and other stems Demucs split from 852
records' middle 40 s (references/stats/_cells/<id>/), embedded into
references/critic/stems/<backbone>/<family>.npz (keys "<id>/drums", ...).

Both are resumable.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
REPO = os.path.dirname(ML)
sys.path[:0] = [REPO, ML, HERE]

import embed  # noqa: E402
from blind_ab import patched  # noqa: E402

REFS = os.path.join(REPO, "references")
TRAIN = os.path.join(REFS, "critic", "train")
STEMS = os.path.join(REFS, "critic", "stems")
PROFILES = ("roots_dub", "steppers_dub", "dubstep_140", "dark_garage", "uk_garage")
OLD = 0.2
# The backbones the critic is trained on: AST won the bake-off (family probe
# 0.65, the lowest panel noise, the most agreement with Ed); CLAP is kept to
# try the two together. LAION's music CLAP did worst and is left out.
USE = ("ast", "clap")


def knobs(prof: str) -> dict[str, list[tuple]]:
    """Each tuned knob and how to put it back as it was before this branch's
    rounds."""
    from riddim.render import audio
    comp = importlib.import_module("riddim.compose")
    fam = {"roots_dub": "dub", "steppers_dub": "dub", "dubstep_140": "dubstep",
           "dark_garage": "dubstep", "uk_garage": "garage"}[prof]
    return {
        "library_off": [("attr", audio, "LIBRARY", False)],
        "tone_off": [("item", audio.TONE, prof, {})],
        "drive_off": [("item", audio.DRIVE, prof, 0.0)],
        "space_off": [("item", audio.SPACE, prof, 1.0)],
        "bubble_full": [("item", audio.BUBBLE_TRIM, prof, 1.0)],
        "bass_mid_full": [("item", audio.BASS_MID, prof, 1.0)],
        "family_gain_off": [("item", audio.FAMILY_GAIN, fam, {})],
        "drummer_off": [("profile", prof, {"drummer": 0.0}), ("item", audio.DRUMMER_MS, prof, 0.0),
                        ("item", audio.DRUMMER_LEAN_MS, prof, 0.0)],
        "rockers_off": [("attr", comp, "ROCKERS_BLOCKS", set())],
        "sixteenths_off": [("profile", prof, {"sub_sixteenths": False})],
        "mined_bass_off": [("profile", prof, {"mined_bass": 0.0})],
        "old_master": [("attr", audio, "MASTER", {"thresh": -14.0, "ratio": 2.5, "makeup": 4.0, "clip": 1.3})],
    }


def lossy(path: str, key: str) -> str:
    """The mix through the codec a record would have come through: the
    references are 86% MP3 and 11% AAC (median 200 kb/s, 128-320), so a
    critic shown clean WAVs against them could learn the codec instead of the
    music. Drawn per song from `key`; returns the encoded file."""
    rng = random.Random(key + ":codec")
    if rng.random() < 0.12:
        out, args = path[:-4] + ".m4a", ["-c:a", "aac", "-b:a", "192k"]
    else:
        kbps = rng.choices((128, 160, 192, 224, 256, 320), (14, 10, 26, 10, 15, 25))[0]
        out, args = path[:-4] + ".mp3", ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, *args, out], check=True)
    return out


def _write(path: str, x: np.ndarray, sr: int) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def flush(files: dict, store: dict, rows: list[str]) -> None:
    for b, f in files.items():
        np.savez(f, **store[b])
    with open(os.path.join(TRAIN, "songs.jsonl"), "a") as fh:
        fh.writelines(r + "\n" for r in rows)
    rows.clear()


def cmd_ours(a) -> int:
    from riddim.compose import compose
    from riddim.render import audio
    os.makedirs(TRAIN, exist_ok=True)
    log = os.path.join(TRAIN, "songs.jsonl")
    done = set()
    if os.path.exists(log):
        done = {(r["profile"], r["seed"]) for r in (json.loads(line) for line in open(log))}
    tmp = tempfile.mkdtemp(prefix="critic_")
    try:
        for prof in PROFILES:
            kn = knobs(prof)
            files = {b: os.path.join(TRAIN, b, prof + ".npz") for b in USE}
            for b in USE:
                os.makedirs(os.path.join(TRAIN, b), exist_ok=True)
            store = {b: dict(np.load(f, allow_pickle=True)) if os.path.exists(f) else {} for b, f in files.items()}
            rows: list[str] = []
            for seed in range(a.seed, a.seed + a.per_profile):
                if (prof, seed) in done:
                    continue
                rng = random.Random(f"{prof}:{seed}:knobs")
                old = sorted(k for k in kn if rng.random() < OLD)
                changes = [c for k in old for c in kn[k]]
                with patched(changes):
                    song = compose(prof, seed=seed)
                    stems: dict = {}
                    mix = audio.render(song, stems_out=stems)
                paths = []
                for part, x in [("mix", mix)] + sorted(stems.items()):
                    # A stem at its mix level can peak over full scale: scaled
                    # to fit, its loudness is normalised for the embedding anyway.
                    peak = float(np.max(np.abs(x))) or 1.0
                    p = os.path.join(tmp, f"{prof}_{seed}_{part}.wav")
                    _write(p, x / max(1.0, peak / 0.98), audio.SR)
                    if part == "mix":
                        wav, p = p, lossy(p, f"{prof}:{seed}")
                        os.remove(wav)
                    paths.append((part, p))
                emb = embed.embed_files([p for _, p in paths], os.path.join(tmp, "_cache"), f"{prof}_{seed}", USE)
                for b, d in emb.items():
                    for part, p in paths:
                        store[b][f"{seed}/{part}"] = d[p]
                for _, p in paths:
                    os.remove(p)
                shutil.rmtree(os.path.join(tmp, "_cache"), ignore_errors=True)
                # A song is logged done only once its embeddings are on disk.
                rows.append(json.dumps({"profile": prof, "seed": seed, "old": old, "bpm": song.bpm}))
                print(f"  {prof} {seed} old={old}", flush=True)
                if len(rows) >= 10:
                    flush(files, store, rows)
            flush(files, store, rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def cmd_demucs(a) -> int:
    """Riddim's songs split the way the records were (mine_cells.split:
    htdemucs on the middle 40 s, each stem mono at riddim.listen's rate), so
    the stem critics compare Demucs output with Demucs output, not with
    riddim's clean buffers. The same songs and knobs as `ours` (from its
    log where it has them). Into references/critic/train_demucs/<backbone>/
    <profile>.npz, keys "<seed>/<stem>"; the audio is deleted."""
    import mine_cells
    from riddim.compose import compose
    from riddim.render import audio
    out_dir = os.path.join(REFS, "critic", "train_demucs")
    logged = {}
    if os.path.exists(os.path.join(TRAIN, "songs.jsonl")):
        for r in (json.loads(line) for line in open(os.path.join(TRAIN, "songs.jsonl"))):
            logged[(r["profile"], r["seed"])] = r["old"]
    tmp = tempfile.mkdtemp(prefix="critic_dm_")
    try:
        for prof in PROFILES:
            kn = knobs(prof)
            files = {b: os.path.join(out_dir, b, prof + ".npz") for b in USE}
            for b in USE:
                os.makedirs(os.path.join(out_dir, b), exist_ok=True)
            store = {b: dict(np.load(f, allow_pickle=True)) if os.path.exists(f) else {} for b, f in files.items()}
            for seed in range(a.seed, a.seed + a.per_profile):
                if f"{seed}/other" in store[USE[0]]:
                    continue
                old = logged.get((prof, seed))
                if old is None:
                    rng = random.Random(f"{prof}:{seed}:knobs")
                    old = sorted(k for k in kn if rng.random() < OLD)
                with patched([c for k in old for c in kn[k]]):
                    mix = audio.render(compose(prof, seed=seed))
                wav = os.path.join(tmp, "mix.wav")
                _write(wav, mix, audio.SR)
                enc = lossy(wav, f"{prof}:{seed}")     # the same codec as the mix set's copy
                folder = os.path.join(tmp, "stems")
                mine_cells.split(enc, folder)
                os.remove(enc)
                paths = [os.path.join(folder, st + ".wav") for st in ("drums", "bass", "other")]
                emb = embed.embed_files(paths, os.path.join(tmp, "_cache"), "s", USE)
                for b, d in emb.items():
                    for st, p in zip(("drums", "bass", "other"), paths):
                        store[b][f"{seed}/{st}"] = d[p]
                shutil.rmtree(folder, ignore_errors=True)
                shutil.rmtree(os.path.join(tmp, "_cache"), ignore_errors=True)
                os.remove(wav)
                if (seed - a.seed + 1) % 10 == 0:
                    for b, f in files.items():
                        np.savez(f, **store[b])
                print(f"  {prof} {seed} split", flush=True)
            for b, f in files.items():
                np.savez(f, **store[b])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


def cmd_stems(a) -> int:
    cells = os.path.join(REFS, "stats", "_cells")
    fam_of = {json.loads(line)["id"]: json.loads(line)["family"]
              for line in open(os.path.join(REFS, "stats", "plan.jsonl"))}
    by_fam: dict[str, list[str]] = {}
    for rid in sorted(os.listdir(cells)):
        if rid in fam_of and all(os.path.exists(os.path.join(cells, rid, s + ".wav")) for s in ("drums", "bass", "other")):
            by_fam.setdefault(fam_of[rid], []).append(rid)
    for fam, ids in by_fam.items():
        paths = [os.path.join(cells, rid, s + ".wav") for rid in ids for s in ("drums", "bass", "other")]
        print(f"{fam}: {len(ids)} records' stems", flush=True)
        emb = embed.embed_files(paths, STEMS, fam, USE)
        del emb
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("ours", "stems", "demucs"))
    ap.add_argument("--per-profile", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args(argv)
    return {"ours": cmd_ours, "stems": cmd_stems, "demucs": cmd_demucs}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())

"""Train the critic: real records against riddim's songs, mix and stems.

    python ml/critic/train.py fit       # train, test, save; writes references/critic/train.md
    python ml/critic/train.py check     # shortcut checks on degraded copies

**fit.** For each backbone (AST, CLAP, and the two side by side):
- *mix*: every reference record's windows (label real) against every training
  song's mix windows (label riddim; ml/critic/dataset.py).
- *drums, bass, other*: the records' Demucs stems against riddim's own stems.

The split is by recording, never by window: 20% of records and 20% of songs
are held out. Two models are tried, a logistic regression and a small MLP
(256 hidden units). A song's score is the mean of its windows' log-odds, and
P(real) is the logistic of that. The held-out AUC is reported per song. The
best model per part is refitted on everything and saved to
references/critic/models/<part>.pkl.

**check.** P(real) for held-out records and for fresh riddim songs (seeds
5000+), each as it is and degraded:
- band-limited to 11 kHz
- one second of silence at the start
- 10 dB quieter
- riddim's: MP3-encoded at 192 kb/s, as the records are; the records': a
  lossless WAV copy, and a second MP3 pass at 128 kb/s

A critic that has learned real music should hardly move on any of these. One
that leans on a shortcut (bandwidth, codec, level, a silent start) will.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
REPO = os.path.dirname(ML)
sys.path[:0] = [REPO, ML, HERE]

import embed  # noqa: E402

REFS = os.path.join(REPO, "references")
CRIT = os.path.join(REFS, "critic")
MODELS = os.path.join(CRIT, "models")
PARTS = ("mix", "drums", "bass", "other")
PROFILES = ("roots_dub", "steppers_dub", "dubstep_140", "dark_garage", "uk_garage")
BACKBONE = "ast+clap"      # set by fit's result; score() reads the saved choice


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _load(path: str) -> dict:
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}


def _feats(parts_by_b: dict[str, np.ndarray], backbone: str) -> np.ndarray:
    """Window features for a backbone name, "a+b" for two side by side."""
    names = backbone.split("+")
    arrs = [parts_by_b[n] for n in names]
    if not all(len(a) for a in arrs):
        return np.zeros((0, 1))
    n = min(len(a) for a in arrs)
    return np.concatenate([_unit(a[:n]) for a in arrs], axis=1)


def collect(part: str, backbone: str, clean: bool = False) -> tuple[list[np.ndarray], list[int], list[str]]:
    """(window arrays, labels, recording ids) for one part: 1 = a record,
    0 = riddim."""
    bbs = backbone.split("+")
    X, y, ids = [], [], []
    if part == "mix":
        for fam_file in sorted(os.listdir(os.path.join(CRIT, bbs[0]))):
            if not fam_file.endswith(".npz"):
                continue
            per_b = {b: _load(os.path.join(CRIT, b, fam_file)) for b in bbs}
            for k in per_b[bbs[0]]:
                if all(k in per_b[b] for b in bbs):
                    f = _feats({b: per_b[b][k] for b in bbs}, backbone)
                    if len(f):
                        X.append(f), y.append(1), ids.append("rec:" + k)
    else:
        for fam_file in sorted(os.listdir(os.path.join(CRIT, "stems", bbs[0]))):
            per_b = {b: _load(os.path.join(CRIT, "stems", b, fam_file)) for b in bbs}
            for k in per_b[bbs[0]]:
                if f"/{part}.wav::" in k and all(k in per_b[b] for b in bbs):
                    f = _feats({b: per_b[b][k] for b in bbs}, backbone)
                    if len(f):
                        X.append(f), y.append(1), ids.append("rec:" + k.split(f"/{part}.wav")[0])
    # The stems: riddim's songs split by Demucs as the records' were
    # (dataset.py demucs) - not its clean buffers, which would teach the
    # critic "Demucs artifacts" instead of "real music".
    src = "train" if part == "mix" or clean else "train_demucs"
    for prof in PROFILES:
        per_b = {b: _load(os.path.join(CRIT, src, b, prof + ".npz")) for b in bbs}
        for k in per_b[bbs[0]]:
            if k.endswith("/" + part) and all(k in per_b[b] for b in bbs):
                f = _feats({b: per_b[b][k] for b in bbs}, backbone)
                if len(f):
                    X.append(f), y.append(0), ids.append(f"ours:{prof}:{k.split('/')[0]}")
    return X, y, ids


def _model(kind: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.1, class_weight="balanced"))
    return make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=1e-3, early_stopping=True,
                                                         max_iter=200, random_state=0))


def _song_logit(clf, X: list[np.ndarray]) -> np.ndarray:
    """A recording's score: the mean of its windows' log-odds."""
    out = []
    for x in X:
        p = np.clip(clf.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
        out.append(float(np.mean(np.log(p / (1 - p)))))
    return np.array(out)


def _balanced_windows(X, y):
    """Windows and labels, each class weighted to the same total, by
    repeating the smaller class's recordings (MLP has no class weights)."""
    Xw = np.concatenate(X)
    yw = np.concatenate([[l] * len(x) for x, l in zip(X, y)])
    n1, n0 = (yw == 1).sum(), (yw == 0).sum()
    if n1 and n0 and abs(n1 - n0) / max(n1, n0) > 0.1:
        small = 1 if n1 < n0 else 0
        reps = int(round(max(n1, n0) / min(n1, n0)))
        idx = np.where(yw == small)[0]
        Xw = np.concatenate([Xw] + [Xw[idx]] * (reps - 1))
        yw = np.concatenate([yw] + [yw[idx]] * (reps - 1))
    return Xw, yw


def cmd_fit(a) -> int:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    os.makedirs(MODELS, exist_ok=True)
    lines = ["# The critic", "", "Held-out AUC per recording (20% of records and of riddim's songs, by "
             "recording), real records against riddim. Mean of window log-odds per recording.", "",
             "| part | backbone | records / riddim songs | logistic AUC | MLP AUC |", "|---|---|---|---|---|"]
    chosen = {}
    for part in PARTS:
        best = None
        for backbone in ("ast", "clap", "ast+clap"):
            X, y, ids = collect(part, backbone)
            y = np.array(y)
            if (y == 1).sum() < 20 or (y == 0).sum() < 20:
                continue
            test = rng.random(len(ids)) < 0.2
            res = {}
            for kind in ("logistic", "mlp"):
                tr = [i for i in range(len(ids)) if not test[i]]
                Xw, yw = _balanced_windows([X[i] for i in tr], y[tr])
                clf = _model(kind).fit(Xw, yw)
                te = [i for i in range(len(ids)) if test[i]]
                res[kind] = float(roc_auc_score(y[te], _song_logit(clf, [X[i] for i in te])))
            lines.append(f"| {part} | {backbone} | {(y == 1).sum()} / {(y == 0).sum()} | "
                         f"{res['logistic']:.3f} | {res['mlp']:.3f} |")
            print(lines[-1], flush=True)
            for kind, auc in res.items():
                if best is None or auc > best[0]:
                    best = (auc, backbone, kind)
        if best is None:
            continue
        auc, backbone, kind = best
        X, y, _ = collect(part, backbone)
        Xw, yw = _balanced_windows(X, np.array(y))
        clf = _model(kind).fit(Xw, yw)
        with open(os.path.join(MODELS, part + ".pkl"), "wb") as f:
            pickle.dump({"model": clf, "backbone": backbone, "kind": kind, "test_auc": auc}, f)
        chosen[part] = (backbone, kind, auc)
    # How far Demucs alone moves riddim's stems: the same songs, clean
    # buffers against Demucs output. Near 0.5 would mean Demucs changes little.
    lines += ["", "Demucs's own mark: riddim's clean stems against the same songs' Demucs stems "
              "(held-out AUC; 0.5 = no difference)", "", "| stem | AUC |", "|---|---|"]
    for part in ("drums", "bass", "other"):
        Xc, _, idc = collect(part, "ast", clean=True)
        Xd, _, idd = collect(part, "ast")
        common = sorted(set(i for i in idc if i.startswith("ours")) & set(i for i in idd if i.startswith("ours")))
        if len(common) < 20:
            continue
        mc = {i: x for i, x in zip(idc, Xc)}
        md = {i: x for i, x in zip(idd, Xd)}
        test = {i for i in common if rng.random() < 0.3}
        X = [mc[i] for i in common] + [md[i] for i in common]
        y = np.array([0] * len(common) + [1] * len(common))
        tr = [j for j, i in enumerate(common * 2) if i not in test]
        te = [j for j, i in enumerate(common * 2) if i in test]
        clf = _model("logistic").fit(*_balanced_windows([X[j] for j in tr], y[tr]))
        lines.append(f"| {part} | {roc_auc_score(y[te], _song_logit(clf, [X[j] for j in te])):.3f} |")
    lines += ["", "Chosen (refitted on everything): "
              + "; ".join(f"{p} {b} {k} (held-out AUC {u:.3f})" for p, (b, k, u) in chosen.items()) + "."]
    with open(os.path.join(CRIT, "train.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

_LOADED: dict = {}


def model(part: str) -> dict:
    if part not in _LOADED:
        with open(os.path.join(MODELS, part + ".pkl"), "rb") as f:
            _LOADED[part] = pickle.load(f)
    return _LOADED[part]


def score_files(paths: dict[str, str], cache_dir: str) -> dict[str, float]:
    """P(real) for each part's audio file ({part: path}), via the saved
    critics."""
    out = {}
    for part, path in paths.items():
        m = model(part)
        bbs = m["backbone"].split("+")
        emb = embed.embed_files([path], cache_dir, part, bbs)
        f = _feats({b: emb[b][path] for b in bbs}, m["backbone"])
        if not len(f):
            continue
        z = float(_song_logit(m["model"], [f])[0])
        out[part] = 1.0 / (1.0 + np.exp(-z))
    return out


def _ffmpeg(src: str, dst: str, *args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *args, dst], check=True)


def cmd_check(a) -> int:
    """P(real) on held-out records and fresh riddim songs, as they are and
    degraded."""
    from riddim.compose import compose
    from riddim.render import audio
    plan = [json.loads(line) for line in open(os.path.join(REFS, "stats", "plan.jsonl"))]
    rng = np.random.default_rng(1)
    recs = [os.path.join(REFS, p["file"]) for p in plan if p["kind"] == "record"]
    seen = set()
    for f in os.listdir(CRIT):
        if f in ("ast", "clap"):
            for g in os.listdir(os.path.join(CRIT, f)):
                seen |= {k.split("::")[0] for k in _load(os.path.join(CRIT, f, g))}
    unseen = [r for r in recs if r not in seen and os.path.exists(r)]
    pick = list(rng.choice(unseen, min(a.n, len(unseen)), replace=False))
    tmp = tempfile.mkdtemp(prefix="critic_check_")
    ours = []
    for i, prof in enumerate(PROFILES):
        for s in range(5000, 5000 + max(1, a.n // len(PROFILES))):
            p = os.path.join(tmp, f"{prof}_{s}.wav")
            audio.write(compose(prof, seed=s), p)
            ours.append(p)
    edits = {
        "as is": [],
        "11 kHz": ["-af", "lowpass=f=11000"],
        "1 s silence first": ["-af", "adelay=1000|1000"],
        "-10 dB": ["-af", "volume=-10dB"],
    }
    lines = ["| set | edit | median P(real) |", "|---|---|---|"]
    for name, files in (("records (held out)", pick), ("riddim (seeds 5000+)", ours)):
        extra = ([("MP3 192k", ["-b:a", "192k"])] if name.startswith("riddim") else
                 [("WAV copy", ["-c:a", "pcm_s16le"]), ("MP3 128k again", ["-b:a", "128k"])])
        for edit, args in list(edits.items()) + extra:
            ps = []
            for j, f in enumerate(files):
                ext = ".mp3" if edit.startswith("MP3") else ".wav"
                g = os.path.join(tmp, f"d{j}_{abs(hash(edit)) % 10**6}{ext}")
                if args:
                    _ffmpeg(f, g, *args)
                else:
                    g = f
                ps.append(score_files({"mix": g}, os.path.join(tmp, "_c"))["mix"])
            lines.append(f"| {name} | {edit} | {np.median(ps):.3f} |")
            print(lines[-1], flush=True)
    with open(os.path.join(CRIT, "check.md"), "w") as f:
        f.write("# Critic shortcut checks\n\n" + "\n".join(lines) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("fit", "check"))
    ap.add_argument("--n", type=int, default=40, help="check: records and songs to test")
    a = ap.parse_args(argv)
    return {"fit": cmd_fit, "check": cmd_check}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())

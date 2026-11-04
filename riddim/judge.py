"""The judge: of two takes of the same music, which sounds more like a record?

It answers A, B or "can't tell", with a confidence, and says why in words.

How it hears: riddim/ears.py turns each 8 s window into named measurements
(tone, punch, space, width, grit, groove). Each is set against the style's
records (how far, in the records' own spread, the take sits from them), and a
take's score is a weighted sum of those distances. The weights are fitted on
pairs where the answer is known without a listener: a record as released
against the same record with a known flaw on it (riddim/defects.py), across
~950 records from ten families. See ml/judge_fit.py for the fit and the
held-out tests, docs/judge.md for the numbers.

Why it says "can't tell": two judges look at every pair. `dist` scores only
the distance from the records (no direction is better than another); `full`
also learns directions (cleaner, drier, less clipped). Each compares the two
takes window by window and calls only when the gap is past the line where it
was 95% right on held-out records and flaws, and steady across the windows
(t >= 3). A winner needs both to call the same side: on unseen artists that
was right 97.6% of the time, and it never called a 0.5 dB nudge. Below that it
says "lean" (one judge calls) or "can't tell". The old judge's failure was
calling 0.001 gaps that nobody, including it, could hear.

What it is for: sound - levels, tone, drive, space, width, the low end. It
does not judge notes, arrangement or which hook is better; those differences
show up as "can't tell" or as a low-confidence call, and are Ed's.

    from riddim import judge
    v = judge.compare("take_a.wav", "take_b.wav", style="dubstep_140")
    v.winner        # "A", "B" or None
    print(v.text)   # the call and the reasons

    riddim-cli judge a.wav b.wav --style roots_dub
    riddim-cli judge take.wav --style dubstep_140      # one take: where it's off
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import ears

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "judge.json")

# A riddim profile (or a rig template) -> the record families it answers to.
STYLES = {
    "dubstep_140": ["dubstep"],
    "dubby_halftime": ["dub_techno", "dubstep"],
    "dark_garage": ["dark_garage", "grime"],
    "uk_garage": ["uk_garage", "speed_garage"],
    "roots_dub": ["roots_dub", "roots_reggae"],
    "steppers_dub": ["steppers_dub", "digital_dub"],
}


def huber(z):
    """A smooth |z|: quadratic near the records, linear far out, so one wild
    measurement can't outvote the rest."""
    return np.sqrt(1.0 + z * z) - 1.0


@dataclass
class Model:
    names: list
    center: dict            # family -> (n_features,)
    scale: dict
    pooled_center: np.ndarray
    pooled_scale: np.ndarray
    u: np.ndarray           # weight on signed z
    w: np.ndarray           # weight on huber(z), >= 0
    tau: float              # smallest gap it calls
    t_min: float = 3.0
    meta: dict = field(default_factory=dict)

    def ref(self, style: str | list | None):
        fams = STYLES.get(style, [style]) if isinstance(style, str) else (style or [])
        fams = [f for f in fams if f in self.center]
        if not fams:
            return self.pooled_center, self.pooled_scale
        c = np.mean([self.center[f] for f in fams], 0)
        s = np.mean([self.scale[f] for f in fams], 0)
        return c, s

    def z(self, F: np.ndarray, style=None) -> np.ndarray:
        c, s = self.ref(style)
        return (np.atleast_2d(F) - c) / s

    def window_scores(self, F: np.ndarray, style=None) -> np.ndarray:
        z = self.z(F, style)
        return z @ self.u - huber(z) @ self.w

    def contributions(self, F: np.ndarray, style=None) -> np.ndarray:
        """Per feature, its part of the score (mean over windows)."""
        z = self.z(F, style)
        return (z * self.u - huber(z) * self.w).mean(0)

    def to_json(self) -> dict:
        return {"names": self.names, "center": {k: v.tolist() for k, v in self.center.items()},
                "scale": {k: v.tolist() for k, v in self.scale.items()},
                "pooled_center": self.pooled_center.tolist(), "pooled_scale": self.pooled_scale.tolist(),
                "u": self.u.tolist(), "w": self.w.tolist(), "tau": self.tau, "t_min": self.t_min,
                "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        a = np.asarray
        return cls(d["names"], {k: a(v) for k, v in d["center"].items()},
                   {k: a(v) for k, v in d["scale"].items()}, a(d["pooled_center"]), a(d["pooled_scale"]),
                   a(d["u"]), a(d["w"]), float(d["tau"]), float(d.get("t_min", 3.0)), d.get("meta", {}))


@dataclass
class Panel:
    """Two judges that must agree. `dist` scores only distance from the
    records (no direction is better than another); `full` adds a signed term
    per measurement (drier, cleaner, less clipped...). Alone, each is ~94%
    right when it calls on unseen artists; a call both make is ~98% right.
    `full` reports the gap and the reasons."""
    heads: dict             # name -> Model

    @property
    def lead(self) -> Model:
        return self.heads.get("full") or next(iter(self.heads.values()))

    def to_json(self) -> dict:
        return {"heads": {k: m.to_json() for k, m in self.heads.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "Panel":
        if "heads" not in d:
            return cls({"full": Model.from_json(d)})
        return cls({k: Model.from_json(v) for k, v in d["heads"].items()})


_LOADED: Panel | None = None


def load_model(path: str = MODEL) -> Panel:
    global _LOADED
    if _LOADED is None or path != MODEL:
        m = Panel.from_json(json.load(open(path)))
        if any(h.names != ears.NAMES for h in m.heads.values()):
            raise RuntimeError("judge model was fitted on different ears features; refit (ml/judge_fit.py)")
        if path != MODEL:
            return m
        _LOADED = m
    return _LOADED


def _panel(model) -> Panel:
    if model is None:
        return load_model()
    return model if isinstance(model, Panel) else Panel({"full": model})


@dataclass
class Verdict:
    winner: str | None      # "A", "B" or None (can't tell)
    gap: float              # mean window score A - B
    t: float                # gap over its standard error
    confidence: str         # "clear", "lean", "can't tell"
    reasons: list           # (feature in words, A's part - B's part), biggest first
    text: str


def _paired(sa: np.ndarray, sb: np.ndarray):
    n = min(len(sa), len(sb))
    d = sa[:n] - sb[:n]
    gap = float(d.mean())
    # windows overlap by half, so about n/2 of them are independent
    se = float(d.std(ddof=1) / np.sqrt(max(1.0, n / 2))) if n > 1 else float("inf")
    return gap, se


def _head(m: Model, Fa, Fb, style, aligned):
    sa, sb = m.window_scores(Fa, style), m.window_scores(Fb, style)
    if aligned:
        gap, se = _paired(sa, sb)
    else:
        gap = float(sa.mean() - sb.mean())
        se = float(np.sqrt(sa.var(ddof=1) / max(1, len(sa) / 2) + sb.var(ddof=1) / max(1, len(sb) / 2)))
    t = gap / se if se > 0 else float("inf") * np.sign(gap)
    called = abs(gap) >= m.tau and abs(t) >= m.t_min
    return gap, t, called


def compare_features(Fa: np.ndarray, Fb: np.ndarray, style=None, model=None,
                     aligned: bool = True) -> Verdict:
    """The panel's verdict on two takes' window features (ears.windows)."""
    panel = _panel(model)
    heads = {k: _head(m, Fa, Fb, style, aligned) for k, m in panel.heads.items()}
    m = panel.lead
    gap, t, _ = heads["full"] if "full" in heads else next(iter(heads.values()))
    sides = {np.sign(g) for g, _, _ in heads.values()}
    if all(c for _, _, c in heads.values()) and len(sides) == 1:
        winner, conf = ("A" if gap > 0 else "B"), "clear"
    elif any(c for _, _, c in heads.values()) and len(sides) == 1:
        winner, conf = None, "lean"
    else:
        winner, conf = None, "can't tell"
    ca, cb = m.contributions(Fa, style), m.contributions(Fb, style)
    d = ca - cb
    order = np.argsort(-np.abs(d))[:5]
    reasons = [(ears.plain(m.names[i]), float(d[i])) for i in order if abs(d[i]) > 0.05 * (abs(gap) + 1e-9)]
    detail = ", ".join(f"{k} {g:+.2f} (t {tt:+.1f}{', calls' if c else ''})" for k, (g, tt, c) in heads.items())
    if winner:
        head = f"{winner} sounds more like the records: both judges call it ({detail})."
    elif conf == "lean":
        head = f"Can't call it: a lean to {'A' if gap > 0 else 'B'}, but only one judge calls it ({detail})."
    else:
        head = f"Can't tell ({detail})."
    why = [f"  {'A' if v > 0 else 'B'} nearer the records on {name} ({abs(v):.2f})" for name, v in reasons]
    what = ["what changed from A to B:", ears.diff_text(Fa, Fb, 5)]
    return Verdict(winner, gap, t, conf, reasons, "\n".join([head] + why + what))


def compare(a: str, b: str, style: str | None = None, model=None,
            aligned: bool = True, start: float = 0.0, seconds: float | None = None) -> Verdict:
    """Judge two recordings of the same music (aligned: the same bars, as two
    renders of one seed or two takes of one Deluge pattern are)."""
    xa = ears.load(a, start=start, seconds=seconds)
    xb = ears.load(b, start=start, seconds=seconds)
    if aligned:
        n = min(len(xa), len(xb))
        xa, xb = xa[:n], xb[:n]
    return compare_features(ears.windows(xa), ears.windows(xb), style, model, aligned)


def report(path_or_features, style: str | None = None, model=None, top: int = 6) -> str:
    """One take on its own: where it sits off the style's records, worst first,
    in the records' own spread (z) and in words."""
    m = _panel(model).lead
    F = ears.windows(ears.load(path_or_features)) if isinstance(path_or_features, str) else np.atleast_2d(path_or_features)
    z = m.z(F, style).mean(0)
    cost = -m.contributions(F, style)
    order = np.argsort(-cost)[:top]
    med = m.meta.get("record_median", {}).get(style)
    vs = f" (the records' own median {med:+.2f})" if med is not None else ""
    lines = [f"score {m.window_scores(F, style).mean():+.2f} against {style or 'all families'}{vs}",
             ears.describe(F.mean(0)), "furthest from the records:"]
    for i in order:
        lines.append(f"  {ears.plain(m.names[i])}: {'over' if z[i] > 0 else 'under'} by {abs(z[i]):.1f} spreads")
    return "\n".join(lines)


# --- the rig: L6max multitrack stems ------------------------------------------

STEM_LEVELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stem_levels.json")

# An L6max channel's role -> the records' Demucs stem it is measured against.
ROLE_STEM = {"drums": "drums", "kick": "drums", "snare": "drums", "hats": "drums", "perc": "drums",
             "bass": "bass", "sub": "bass", "malevolent": "bass",
             "chords": "other", "skank": "other", "pad": "other", "pads": "other", "lead": "other",
             "keys": "other", "organ": "other", "fx": "other", "delay": "other", "coral": "other",
             "vocal": "vocals", "vocals": "vocals", "voice": "vocals"}


def _level_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-20))


def stem_balance(stems: dict, style: str | None = None, mix: np.ndarray | None = None) -> list:
    """Each stem group's level against the mix, set beside the records'.

    stems: role -> (n, 2) audio, *not* loudness-normalised (ears.load(..., norm=False)),
    all from one take. The mix is their sum unless given (the L6max's own
    master track). Returns rows (stem, ours dB, records' median, q25, q75, verdict)."""
    ref = json.load(open(STEM_LEVELS))
    fams = [f for f in (STYLES.get(style, [style]) if style else []) if f in ref] or \
        [f for f in ref if not f.startswith("_")]
    n = min(len(v) for v in stems.values())
    groups: dict = {}
    for role, x in stems.items():
        g = ROLE_STEM.get(role.lower().rstrip("0123456789_ "), "other")
        groups[g] = groups.get(g, 0) + x[:n]
    total = mix[:n] if mix is not None else sum(groups.values())
    lm = _level_db(total)
    rows = []
    for g in ("drums", "bass", "other", "vocals"):
        med = float(np.mean([ref[f][g]["median"] for f in fams]))
        q25 = float(np.mean([ref[f][g]["q25"] for f in fams]))
        q75 = float(np.mean([ref[f][g]["q75"] for f in fams]))
        if g not in groups:
            rows.append((g, None, med, q25, q75, "none on the rig" if g == "vocals" else "missing"))
            continue
        ours = _level_db(groups[g]) - lm
        v = "in the records' range" if q25 <= ours <= q75 else (
            f"{ours - q75:+.1f} dB over the records' range" if ours > q75 else f"{ours - q25:+.1f} dB under it")
        rows.append((g, ours, med, q25, q75, v))
    return rows


def rig_report(stems: dict, style: str | None = None, mix_path: str | None = None,
               model=None) -> str:
    """What the rig is doing, from one multitrack take: the mix against the
    style's records, the stem balance against theirs, and each channel in
    words. stems: role -> path (e.g. {"drums": "L6_ch1.wav", "bass": "L6_ch3.wav"})."""
    raw = {r: ears.load(p, norm=False) for r, p in stems.items()}
    mix = ears.load(mix_path, norm=False) if mix_path else None
    n = min(len(v) for v in raw.values())
    total = mix[:n] if mix is not None else sum(v[:n] for v in raw.values())
    lines = ["THE MIX", report(ears.windows(ears.normalize(total)), style, model), "", "STEM BALANCE (dB against the mix)"]
    for g, ours, med, q25, q75, v in stem_balance(raw, style, mix):
        o = "  -  " if ours is None else f"{ours:+5.1f}"
        lines.append(f"  {g:7s} ours {o}  records {med:+.1f} ({q25:+.1f} to {q75:+.1f})  {v}")
    lines += ["", "EACH CHANNEL"]
    for r, x in raw.items():
        if _level_db(x) < -90:
            lines.append(f"  {r}: silent")
            continue
        lines.append(f"  {r} (peak {20 * np.log10(np.max(np.abs(x)) + 1e-12):.1f} dBFS, "
                     f"rms {_level_db(x):.1f} dBFS):")
        lines += ["    " + s for s in ears.describe(ears.windows(ears.normalize(x)).mean(0)).splitlines()]
    return "\n".join(lines)

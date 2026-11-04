"""Fit the judge (riddim/judge.py) and test it where the answer is known.

    python ml/judge_fit.py --feats judge_feats.npz [--ed ed_feats.json] [--write]

The material (ml/judge_data.py): reference records' excerpts, each clean and
with known flaws on it (riddim/defects.py), measured per 8 s window by
riddim/ears.py. A pair is a record against itself with one flaw; the record
should win. A "nudge" pair (a 0.5 dB wiggle) should be "can't tell".

Tests, all held out:
1. **unseen artists** - 5 folds by artist; the records' centres and spreads,
   the weights and the calling line are all fitted on the other artists.
2. **unseen flaws** - each flaw group (EQ, bandwidth, dynamics, space, stereo,
   noise) held out in turn: fitted without that group, tested on it, on
   unseen artists. This is the test of a flaw nobody thought to simulate.
3. **the old way** - closeness to the records' centre with no learned
   weights, the analogue of the CLAP judge's closeness, on the same pairs.
4. **Ed's picks** (--ed, from ml/judge_ed.py's feature dump) - the blind A/B
   and the library picks, with the final model.

--write saves the model fitted on everything to riddim/data/judge.json and the
report to docs/judge-tests.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from riddim import defects, ears, judge  # noqa: E402

MODES = ("dist", "full")


def family_stats(F, fam, mask):
    """Robust centre and spread per family from clean windows, shrunk to the
    pooled figures for small families."""
    Fm, fm = F[mask], fam[mask]
    pc = np.median(Fm, 0)
    q75, q25 = np.percentile(Fm, [75, 25], 0)
    ps = np.maximum((q75 - q25) / 1.349, 1e-3)
    floor = 0.25 * ps
    cen, sca = {}, {}
    k = 40.0
    for f in np.unique(fm):
        X = Fm[fm == f]
        n = len(X)
        c = np.median(X, 0)
        a, b = np.percentile(X, [75, 25], 0)
        s = np.maximum((a - b) / 1.349, floor)
        cen[f] = (n * c + k * pc) / (n + k)
        sca[f] = (n * s + k * ps) / (n + k)
    return cen, sca, pc, ps


def zfeats(F, fam, cen, sca, pc, ps):
    C = np.stack([cen.get(f, pc) for f in fam])
    S = np.stack([sca.get(f, ps) for f in fam])
    return (F - C) / S


class Pairs:
    """Window-aligned pairs: clean window index, flawed window index, flaw, size, excerpt."""

    def __init__(self, d):
        self.F = d["F"].astype(np.float64)
        self.fam, self.art, self.exc = d["family"], d["artist"], d["excerpt"]
        self.flaw, self.size, self.win = d["flaw"], d["size"], d["win"]
        clean = {}
        for i in np.where(self.flaw == "clean")[0]:
            clean[(self.exc[i], self.win[i])] = i
        ia, ib = [], []
        for i in np.where(self.flaw != "clean")[0]:
            j = clean.get((self.exc[i], self.win[i]))
            if j is not None:
                ia.append(j)
                ib.append(i)
        self.ia, self.ib = np.array(ia), np.array(ib)
        self.pflaw = self.flaw[self.ib]
        self.psize = self.size[self.ib]
        self.pexc = self.exc[self.ib]
        self.part = self.art[self.ib]
        self.pgroup = np.array(["tie" if f == "nudge" else defects.group(f) for f in self.pflaw])
        self.clean_mask = self.flaw == "clean"


def fit_weights(dz, dhs, lam=1e-2, full=True):
    """Logistic fit of margin = dz.u - sum_b dh_b.w_b (A, the record, should
    win), every w_b >= 0. Returns u and the list of w_b."""
    n, k = dz.shape
    nb = len(dhs)
    ku = k if full else 0

    def f(p):
        u = p[:ku]
        m = (dz @ u if full else 0) - sum(dh @ p[ku + b * k: ku + (b + 1) * k] for b, dh in enumerate(dhs))
        l = np.logaddexp(0, -m).mean() + lam * (p @ p)
        s = -1 / (1 + np.exp(m))
        g = ([dz.T @ s / n] if full else []) + [-(dh.T @ s) / n for dh in dhs]
        return l, np.concatenate(g) + 2 * lam * p

    p0 = np.zeros(ku + nb * k)
    p0[ku:] = 0.01
    bounds = [(None, None)] * ku + [(0, None)] * (nb * k)
    r = minimize(f, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    u = r.x[:ku] if full else np.zeros(k)
    return u, [r.x[ku + b * k: ku + (b + 1) * k] for b in range(nb)]


def scorer(Z, u, ws):
    return Z @ u - judge.huber(Z) @ ws[0]


def fit_mode(P, Z, idx, mode, _unused=None):
    dz = Z[P.ia[idx]] - Z[P.ib[idx]]
    H = judge.huber
    return fit_weights(dz, [H(Z[P.ia[idx]]) - H(Z[P.ib[idx]])], full=(mode == "full"))


def clip_calls(P, Z, idx, u, ws):
    """Per (excerpt, flaw, size) pair in idx: paired gap and t over windows."""
    s = scorer(Z, u, ws)
    groups = defaultdict(list)
    for i in idx:
        groups[(P.pexc[i], P.pflaw[i], P.psize[i])].append(s[P.ia[i]] - s[P.ib[i]])
    out = []
    for (e, fl, sz), d in groups.items():
        d = np.array(d)
        gap = d.mean()
        se = d.std(ddof=1) / np.sqrt(max(1, len(d) / 2)) if len(d) > 1 else np.inf
        out.append((e, fl, sz, gap, gap / se if se > 0 else np.inf))
    return out


def choose_tau(calls, t_min=3.0, target=0.95, max_false=0.02):
    """The smallest gap at which calls on flaws are >= target right and nudges
    are called <= max_false of the time."""
    flaws = [(g, t) for _, fl, _, g, t in calls if fl != "nudge"]
    ties = [(g, t) for _, fl, _, g, t in calls if fl == "nudge"]
    gaps = np.array([abs(g) for g, _ in flaws])
    for tau in np.quantile(gaps, np.linspace(0, 0.95, 96)):
        called = [(g, t) for g, t in flaws if abs(g) >= tau and abs(t) >= t_min]
        if not called:
            continue
        prec = np.mean([g > 0 for g, _ in called])
        fa = np.mean([abs(g) >= tau and abs(t) >= t_min for g, t in ties]) if ties else 0
        if prec >= target and fa <= max_false:
            return float(tau)
    return float(np.max(gaps))


def run_fold(P, train_art, test_art, mode, exclude_group=None, t_min=3.0):
    """One head fitted on train_art (stats, weights, calling line), its calls
    on test_art: {(excerpt, flaw, size): (gap, t, called)}."""
    tr = np.isin(P.art, list(train_art))
    cen, sca, pc, ps = family_stats(P.F, P.fam, tr & P.clean_mask)
    Z = zfeats(P.F, P.fam, cen, sca, pc, ps)
    ptr = np.isin(P.part, list(train_art)) & (P.pgroup != "tie")
    if exclude_group:
        ptr &= P.pgroup != exclude_group
    # an inner split of the training artists sets the calling line
    arts = np.array(sorted(train_art))
    rng = np.random.default_rng(0)
    inner = set(rng.choice(arts, size=max(1, len(arts) // 5), replace=False))
    pin = ptr & ~np.isin(P.part, list(inner))
    u, ws = fit_mode(P, Z, pin, mode, None)
    vidx = np.where(np.isin(P.part, list(inner)) & ((P.pgroup != exclude_group) if exclude_group else True))[0]
    tau = choose_tau(clip_calls(P, Z, vidx, u, ws), t_min)
    u, ws = fit_mode(P, Z, ptr, mode, None)
    tidx = np.where(np.isin(P.part, list(test_art)))[0]
    if exclude_group:
        tidx = tidx[(P.pgroup[tidx] == exclude_group) | (P.pgroup[tidx] == "tie")]
    return {(e, fl, sz): (g, t, abs(g) >= tau and abs(t) >= t_min)
            for e, fl, sz, g, t in clip_calls(P, Z, tidx, u, ws)}, tau


def consensus(heads):
    """Both heads' calls -> the panel's: a winner only when both call it."""
    keys = next(iter(heads.values())).keys()
    out = {}
    for k in keys:
        vals = [h[k] for h in heads.values()]
        same = len({np.sign(g) for g, _, _ in vals}) == 1
        out[k] = (vals[-1][0], vals[-1][1], same and all(c for _, _, c in vals))
    return out


def tally(calls):
    """calls -> {(flaw, size): [n, called, right]}."""
    by = defaultdict(lambda: [0, 0, 0])
    for (e, fl, sz), (g, t, c) in calls.items():
        v = by[(fl, int(sz))]
        v[0] += 1
        v[1] += c
        v[2] += c and g > 0
    return by


def merge(bys):
    out = defaultdict(lambda: [0, 0, 0])
    for by in bys:
        for k, v in by.items():
            for j in range(3):
                out[k][j] += v[j]
    return out


def agg(by, pred=lambda fl: fl != "nudge"):
    n = sum(v[0] for k, v in by.items() if pred(k[0]))
    c = sum(v[1] for k, v in by.items() if pred(k[0]))
    r = sum(v[2] for k, v in by.items() if pred(k[0]))
    return n, c, r


def baseline_calls(P, train_art, test_art):
    """The old way: nearer the records' centre (unweighted distance in spreads) wins, always."""
    tr = np.isin(P.art, list(train_art))
    cen, sca, pc, ps = family_stats(P.F, P.fam, tr & P.clean_mask)
    Z = zfeats(P.F, P.fam, cen, sca, pc, ps)
    k = Z.shape[1]
    tidx = np.where(np.isin(P.part, list(test_art)))[0]
    return {(e, fl, sz): (g, t, True) for e, fl, sz, g, t in clip_calls(P, Z, tidx, np.zeros(k), [np.ones(k) / k])}


def folds(P, k=5):
    arts = np.array(sorted(set(P.art)))
    rng = np.random.default_rng(1)
    rng.shuffle(arts)
    return [set(arts[i::k]) for i in range(k)]


def wilson(r, n, z=1.96):
    """95% interval on a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = r / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def line(label, by):
    n, c, r = agg(by)
    tn, tc, _ = agg(by, lambda fl: fl == "nudge")
    lo, hi = wilson(r, c)
    return (f"| {label} | {c}/{n} ({c / max(n, 1):.0%}) | **{r / max(c, 1):.1%}** ({lo:.1%}-{hi:.1%}) | "
            f"{tc}/{tn} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--ed")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    d = np.load(a.feats, allow_pickle=False)
    P = Pairs(d)
    print(f"{len(P.ia)} window pairs, {len(set(P.pexc))} excerpts, {len(set(P.art))} artists", flush=True)
    F5 = folds(P)
    allart = set(P.art)
    cv = {m: [] for m in MODES}
    taus = {m: [] for m in MODES}
    logo = {g: {m: [] for m in MODES} for g in defects.GROUPS}
    for i, te in enumerate(F5):
        for m in MODES:
            c, tau = run_fold(P, allart - te, te, m)
            cv[m].append(c)
            taus[m].append(tau)
            for g in defects.GROUPS:
                logo[g][m].append(run_fold(P, allart - te, te, m, exclude_group=g)[0])
        print(f"fold {i + 1}/5", flush=True)
    by = {m: merge(tally(c) for c in cv[m]) for m in MODES}
    by["panel"] = merge(tally(consensus({m: cv[m][i] for m in MODES})) for i in range(len(F5)))
    by_logo = {g: {"panel": merge(tally(consensus({m: logo[g][m][i] for m in MODES})) for i in range(len(F5))),
                   **{m: merge(tally(c) for c in logo[g][m]) for m in MODES}} for g in defects.GROUPS}
    base = merge(tally(baseline_calls(P, allart - te, te)) for te in F5)

    cen, sca, pc, ps = family_stats(P.F, P.fam, P.clean_mask)
    Z = zfeats(P.F, P.fam, cen, sca, pc, ps)
    pall = P.pgroup != "tie"
    heads = {}
    for m in MODES:
        u, ws = fit_mode(P, Z, pall, m, None)
        heads[m] = judge.Model(list(ears.NAMES), cen, sca, pc, ps, u, ws[0], float(np.median(taus[m])), 3.0,
                               meta={"records": int(len(set(d["rec"]))), "artists": len(allart), "head": m,
                                     "window_pairs": int(len(P.ia))})
    for m in MODES:     # where the records themselves score, per style, for single-take reports
        h = heads[m]
        h.meta["record_median"] = {}
        for style, fams in judge.STYLES.items():
            mask = P.clean_mask & np.isin(P.fam, fams)
            h.meta["record_median"][style] = float(np.median(h.window_scores(P.F[mask], style)))
    panel = judge.Panel(heads)
    ed_rows = ed_check(panel, json.load(open(a.ed))) if a.ed else []
    report = render_report(d, panel, by, by_logo, base, ed_rows)
    print(report)
    if a.write:
        os.makedirs(os.path.dirname(judge.MODEL), exist_ok=True)
        with open(judge.MODEL, "w") as f:
            json.dump(panel.to_json(), f, separators=(",", ":"))
        with open(os.path.join(REPO, "docs", "judge-tests.md"), "w") as f:
            f.write(report)
        print("wrote", judge.MODEL, "and docs/judge-tests.md")
    return 0


def ed_check(panel, pairs):
    rows = ["| set | # | profile | change | Ed picked | judge | gaps (dist / full) | agrees |",
            "|---|---|---|---|---|---|---|---|"]
    tally_ = defaultdict(int)
    for p in pairs:
        v = judge.compare_features(np.array(p["wa"]), np.array(p["wb"]), p["profile"], panel)
        gd = judge._head(panel.heads["dist"], np.array(p["wa"]), np.array(p["wb"]), p["profile"], True)[0]
        lean = "A" if v.gap > 0 else "B"
        ed = {"kept": "A", "other": "B", None: "can't tell"}[p["ed"]]
        said = v.winner or ("lean " + lean if v.confidence == "lean" else "can't tell")
        if p["ed"] is None:
            ok = "yes" if v.winner is None else "**no**"
        elif v.winner is None:
            ok = "abstained"
        else:
            ok = "yes" if v.winner == ed else "**no**"
        tally_[ok] += 1
        rows.append(f"| {p['src']} | {p['id']} | {p['profile']} | {p['change']} | {ed} | {said} | "
                    f"{gd:+.2f} / {v.gap:+.2f} | {ok} |")
    rows.append("")
    rows.append("Totals: " + ", ".join(f"{k} {v}" for k, v in sorted(tally_.items())) + ".")
    return rows


def render_report(d, panel, by, by_logo, base, ed_rows):
    m0 = panel.lead
    L = ["# The judge, tested", "",
         "Generated by `ml/judge_fit.py`. Material: "
         f"{m0.meta['records']} records by {m0.meta['artists']} artists in {len(set(d['family']))} families, "
         f"two 20 s excerpts each, {m0.meta['window_pairs']} window pairs. A flaw pair is a record against itself "
         "with one known flaw (riddim/defects.py); the record should win. A nudge pair (a 0.5 dB wiggle) should be "
         "can't tell. Right = the record won, of the pairs it called; the interval is 95% (Wilson).", "",
         "## Unseen artists (5 folds by artist)", "",
         "| judge | called | right when called | nudges called |", "|---|---|---|---|",
         line("panel (both must call)", by["panel"]), line("dist alone", by["dist"]), line("full alone", by["full"])]
    bn, _, br = agg(base)
    L.append(f"| the old way: nearer the records' centre, always calls | {bn}/{bn} (100%) | {br / bn:.1%} | - |")
    L += ["", "## Unseen flaw types (each group left out of the fit, tested on unseen artists)", "",
          "| group held out | judge | called | right when called | nudges called |", "|---|---|---|---|---|"]
    for g, b in by_logo.items():
        for m in ("panel", "dist", "full"):
            L.append(line(f"{g} | {m}", b[m]))
    L += ["", "## The panel by flaw and size (unseen artists)", "",
          "| flaw | group | size | pairs | called | right when called |", "|---|---|---|---|---|---|"]
    sizes = {nm: s for nm, _, s in defects.CATALOGUE}
    for (fl, sz), (n, c, r) in sorted(by["panel"].items(), key=lambda kv: (kv[0][0] != "nudge", kv[0])):
        g = "tie" if fl == "nudge" else defects.group(fl)
        size = sizes[fl][sz] if fl in sizes else 0.5
        L.append(f"| {fl} | {g} | {size} | {n} | {c / n:.0%} | {(f'{r / c:.1%}' if c else '-')} |")
    if ed_rows:
        L += ["", "## Ed's picks (the final panel; it never saw them)", "", *ed_rows]
    L += ["", "## What the full judge weighs most", "",
          "| measurement | weight on distance from the records | signed weight (+ = more is better) |", "|---|---|---|"]
    f = panel.heads["full"]
    for i in np.argsort(-(f.w + np.abs(f.u)))[:15]:
        L.append(f"| {ears.plain(f.names[i])} | {f.w[i]:.3f} | {f.u[i]:+.3f} |")
    L.append("")
    L.append("Calling lines: " + ", ".join(f"{k} gap >= {h.tau:.2f} and t >= {h.t_min:.0f}" for k, h in panel.heads.items()) + ".")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())

"""A blind A/B of the judge's own decisions, for Ed's ears: does the judge
pick what he picks? The plan's gate for trusting it.

    python ml/blind_ab.py build            # render the pairs (about 10 minutes)
    python ml/blind_ab.py serve            # open the listening page
    python ml/blind_ab.py score            # Ed's picks against the judge's

Each pair is one past keep-or-revert call made on the goal panel: the same
song (one seed) rendered as the branch has it and with the change undone, or
with the rejected version. Both sides are cut to the same span of about 35 s,
starting two bars before the first drop, and matched in loudness (RMS, with a
shared peak ceiling), so loudness alone cannot pick. The page shuffles the
pairs and which side is A on every load, and plays nothing that names a side.
Picks go to ab-blind/picks.jsonl, one line each, with "can't tell" allowed.

The judge's verdict for a pair is the call it made on the panel ("kept"):
the side the branch has now, or the side it would have kept had the change
won. score.py also reports the judge's per-clip closeness to the family's
references, so a mismatch can be read either way.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
import sys
import webbrowser
from dataclasses import replace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, HERE]

OUT = os.path.join(REPO, "ab-blind")
CLIP_SECONDS = 35.0
TARGET_RMS_DB = -16.0


@contextlib.contextmanager
def patched(changes: list[tuple]):
    """Set module attributes, dict entries and profile fields for one render,
    then put them back. Each change is ("attr", module, name, value),
    ("item", mapping, key, value), ("profile", name, {field: value}) or
    ("env", name, value)."""
    from riddim import profile
    undo = []
    try:
        for c in changes:
            if c[0] == "attr":
                _, mod, name, value = c
                undo.append(("attr", mod, name, getattr(mod, name)))
                setattr(mod, name, value)
            elif c[0] == "item":
                _, mapping, key, value = c
                undo.append(("item", mapping, key, mapping.get(key, KeyError)))
                if value is KeyError:
                    mapping.pop(key, None)
                else:
                    mapping[key] = value
            elif c[0] == "profile":
                _, name, fields = c
                undo.append(("profile", name, profile.PROFILES[name]))
                profile.PROFILES[name] = replace(profile.PROFILES[name], **fields)
            elif c[0] == "env":
                _, name, value = c
                undo.append(("env", name, os.environ.get(name)))
                os.environ[name] = value
        yield
    finally:
        for u in reversed(undo):
            if u[0] == "attr":
                setattr(u[1], u[2], u[3])
            elif u[0] == "item":
                if u[3] is KeyError:
                    u[1].pop(u[2], None)
                else:
                    u[1][u[2]] = u[3]
            elif u[0] == "profile":
                profile.PROFILES[u[1]] = u[2]
            elif u[0] == "env":
                if u[2] is None:
                    os.environ.pop(u[1], None)
                else:
                    os.environ[u[1]] = u[2]


def pairs() -> list[dict]:
    """The calls, each as the judge made it: `kept` is the side it chose,
    `other` the side it turned down. Clear wins and close calls, across
    families."""
    from riddim.render import audio
    comp = importlib.import_module("riddim.compose")
    return [
        dict(profile="roots_dub", seed=302, change="a drummer: touches and a player's timing", call="clear",
             kept=[], other=[("profile", "roots_dub", {"drummer": 0.0}), ("attr", audio, "DRUMMER_MS", {})]),
        dict(profile="roots_dub", seed=303, change="every other four bars as rockers", call="close",
             kept=[], other=[("attr", comp, "ROCKERS_BLOCKS", set())]),
        dict(profile="roots_dub", seed=304, change="organ bubble 4 dB down", call="close",
             kept=[], other=[("item", audio.BUBBLE_TRIM, "roots_dub", KeyError)]),
        dict(profile="roots_dub", seed=305, change="bass guitar's mid at half", call="close",
             kept=[], other=[("item", audio.BASS_MID, "roots_dub", KeyError)]),
        dict(profile="steppers_dub", seed=306, change="dub drums 2 dB down", call="close",
             kept=[], other=[("item", audio.FAMILY_GAIN, "dub", KeyError)]),
        dict(profile="uk_garage", seed=301, change="reverb and delay x2.5", call="clear",
             kept=[], other=[("item", audio.SPACE, "uk_garage", KeyError)]),
        dict(profile="uk_garage", seed=302, change="short chord stabs, chords 4.4 dB down", call="close",
             kept=[], other=[("profile", "uk_garage", {"skank_length": 1.0}),
                             ("item", audio.FAMILY_GAIN, "garage", KeyError)]),
        dict(profile="uk_garage", seed=303, change="Rhodes instead of organ", call="split",
             kept=[], other=[("env", "RIDDIM_PICKS", "garage.organ=Soulful Keys/Multisamples/Organ0[1234]/")]),
        dict(profile="uk_garage", seed=304, change="master drive to the records' loudness", call="clear",
             kept=[], other=[("item", audio.DRIVE, "uk_garage", 0.0)]),
        dict(profile="dubstep_140", seed=305, change="master drive to the records' loudness", call="clear",
             kept=[], other=[("item", audio.DRIVE, "dubstep_140", 0.0)]),
        dict(profile="dark_garage", seed=306, change="electric piano instead of the dub organ", call="clear",
             kept=[], other=[("env", "RIDDIM_PICKS", "dubstep.organ=Soulful Keys/Multisamples/Organ0[34]/")]),
        dict(profile="dubstep_140", seed=307, change="140 drums 2 dB down (rejected)", call="close",
             kept=[], other=[("item", audio.FAMILY_GAIN, "dubstep", {"drums": 0.8})]),
    ]


def _span(song) -> tuple[int, int]:
    """About CLIP_SECONDS from two bars before the first drop (energy 0.8 or
    more), else from the middle."""
    from riddim.units import TICKS_PER_BAR
    secs = song.arrangement.sections
    bar_s = 4 * 60.0 / song.bpm
    bars = max(4, int(round(CLIP_SECONDS / bar_s)))
    drop = next((s for s in secs if s.energy >= 0.8), None)
    start = max(0, (drop.start_bar - 2) if drop else song.bars // 2 - bars // 2)
    start = min(start, max(0, song.bars - bars))
    return start * TICKS_PER_BAR, (start + bars) * TICKS_PER_BAR


def _match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both sides at the same RMS, under one shared peak ceiling."""
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    target = 10 ** (TARGET_RMS_DB / 20)
    a *= target / (np.sqrt(np.mean(a ** 2)) or 1.0)
    b *= target / (np.sqrt(np.mean(b ** 2)) or 1.0)
    peak = max(np.max(np.abs(a)), np.max(np.abs(b)))
    if peak > 0.98:
        a, b = a * 0.98 / peak, b * 0.98 / peak
    from riddim.render.audio import SR
    fade = np.linspace(0, 1, int(0.02 * SR))[:, None]
    for x in (a, b):
        x[: len(fade)] *= fade
        x[-len(fade):] *= fade[::-1]
    return a.astype(np.float32), b.astype(np.float32)


def _write(path: str, x: np.ndarray) -> None:
    import wave
    from riddim.render.audio import SR
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def cmd_build(a) -> int:
    import judge
    import goal_check as gc
    from riddim.compose import compose
    from riddim.render import audio
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    meta = []
    for i, p in enumerate(pairs()):
        sides = {}
        for side in ("kept", "other"):
            with patched(p[side]):
                song = compose(p["profile"], seed=p["seed"])
                span = _span(song)
                sides[side] = audio.render(song, span)
        k, o = _match(sides["kept"], sides["other"])
        files = {}
        for side, x in (("kept", k), ("other", o)):
            # Neutral names: nothing in a file name says which side it is.
            name = f"clips/{i:02d}_{random.Random(f'{i}:{side}:{a.salt}').getrandbits(32):08x}.wav"
            _write(os.path.join(a.out, name), x)
            files[side] = name
        meta.append({"id": i, "profile": p["profile"], "seed": p["seed"], "change": p["change"],
                     "call": p["call"], "files": files})
        print(f"  {i:2d} {p['profile']:13s} {p['seed']} {p['change']}", flush=True)
    # The judge on the clips themselves: closeness to the family's references.
    emb = judge.embed_many([os.path.join(a.out, m["files"][s]) for m in meta for s in ("kept", "other")],
                           os.path.join(a.out, "clips_emb.npz"))
    cents = gc.centroids(os.path.join(REPO, "references", "judge"))
    for m in meta:
        fam = gc.families_for(m["profile"])[0]
        for s in ("kept", "other"):
            e = judge._unit(emb[os.path.join(a.out, m["files"][s])])
            m.setdefault("clip_closeness", {})[s] = round(float((e @ cents[fam]).mean()), 4)
    # The page gets only what it needs to play and to send a pick back.
    with open(os.path.join(a.out, "pairs.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(a.out, "index.html"), "w") as f:
        f.write(PAGE)
    print(f"{len(meta)} pairs -> {a.out}  (now: python ml/blind_ab.py serve)")
    return 0


class _Handler(SimpleHTTPRequestHandler):
    out = OUT

    def __init__(self, *args, **kw):
        super().__init__(*args, directory=self.out, **kw)

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.split("?")[0] == "/pairs":
            meta = json.load(open(os.path.join(self.out, "pairs.json")))
            # Sides as opaque ids; the page decides A and B.
            body = json.dumps([{"id": m["id"], "clips": [m["files"]["kept"], m["files"]["other"]]}
                               for m in meta]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split("?")[0].endswith((".json", ".jsonl", ".npz")):
            self.send_error(404)       # the key and the picks stay off the page
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/pick":
            self.send_error(404)
            return
        rec = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with open(os.path.join(self.out, "picks.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.send_response(204)
        self.end_headers()


def cmd_serve(a) -> int:
    if not os.path.exists(os.path.join(a.out, "pairs.json")):
        print("no pairs yet: python ml/blind_ab.py build", file=sys.stderr)
        return 1
    _Handler.out = a.out
    server = ThreadingHTTPServer(("127.0.0.1", a.port), _Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"blind A/B at {url}  (picks -> {os.path.join(a.out, 'picks.jsonl')}; Ctrl-C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_score(a) -> int:
    from math import comb
    meta = {m["id"]: m for m in json.load(open(os.path.join(a.out, "pairs.json")))}
    path = os.path.join(a.out, "picks.jsonl")
    if not os.path.exists(path):
        print("no picks yet")
        return 1
    last = {}
    for line in open(path):
        r = json.loads(line)
        last[r["id"]] = r                      # a re-pick replaces the earlier one
    agree = disagree = unsure = 0
    print("| pair | profile | change | call | Ed picked | judge kept | clip closeness kept / other |")
    print("|---|---|---|---|---|---|---|")
    for i, m in sorted(meta.items()):
        r = last.get(i)
        if r is None:
            pick = "-"
        elif r["pick"] == "unsure":
            pick, unsure = "can't tell", unsure + 1
        else:
            side = "kept" if r["clip"] == m["files"]["kept"] else "other"
            pick = side
            agree += side == "kept"
            disagree += side == "other"
        cc = m.get("clip_closeness", {})
        print(f"| {i} | {m['profile']} | {m['change']} | {m['call']} | {pick} | kept | "
              f"{cc.get('kept', 0):.3f} / {cc.get('other', 0):.3f} |")
    n = agree + disagree
    if n:
        # One-sided sign test: the chance of agreeing this often by coin flip.
        p = sum(comb(n, k) for k in range(agree, n + 1)) / 2 ** n
        print(f"\nEd agrees with the judge on {agree} of {n} decided pairs ({agree / n:.0%}), "
              f"{unsure} can't tell; by chance alone p = {p:.3f}")
    return 0


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind A/B</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a66;--line:#dddad3;--pick:#1f6f4a;--pickbg:#e3f1e9}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--card:#1f1f1d;--ink:#ecebe7;--mute:#9a9893;--line:#33322f;--pick:#7fd1a6;--pickbg:#1e3329}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}.lead{color:var(--mute);margin:0 0 18px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:12px}
h2{font-size:16px;margin:0 0 10px}.row{display:grid;grid-template-columns:28px 1fr;gap:10px;align-items:center;margin-bottom:8px}
.row b{font-size:15px}audio{width:100%}.picks{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}
button{font:inherit;padding:7px 14px;border-radius:7px;border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
button.on{background:var(--pickbg);border-color:var(--pick);color:var(--pick);font-weight:600}#done{color:var(--mute)}
</style></head><body><main>
<h1>Blind A/B</h1>
<p class="lead">Each pair is the same moment of one song, made two ways and matched in loudness. Pick the one that sounds more like a record. If they sound the same, say so. Your picks are saved as you go.</p>
<p id="done"></p><div id="pairs"></div>
</main><script>
const shuffle=(a)=>{for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a};
fetch("/pairs").then(r=>r.json()).then(pairs=>{
  const root=document.getElementById("pairs");let picked=0;
  shuffle(pairs).forEach((p,n)=>{
    const clips=shuffle(p.clips.slice());
    const el=document.createElement("section");
    el.innerHTML=`<h2>Pair ${n+1} of ${pairs.length}</h2>
      <div class="row"><b>A</b><audio controls preload="none" src="${clips[0]}"></audio></div>
      <div class="row"><b>B</b><audio controls preload="none" src="${clips[1]}"></audio></div>
      <div class="picks"><button data-v="0">A</button><button data-v="1">B</button><button data-v="u">can't tell</button></div>`;
    const btns=el.querySelectorAll("button");
    btns.forEach(b=>b.onclick=()=>{
      const first=![...btns].some(x=>x.classList.contains("on"));
      btns.forEach(x=>x.classList.toggle("on",x===b));
      const v=b.dataset.v;
      fetch("/pick",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({id:p.id,pick:v==="u"?"unsure":"clip",clip:v==="u"?null:clips[+v],at:new Date().toISOString()})});
      if(first){picked++;document.getElementById("done").textContent=`${picked} of ${pairs.length} picked`;}
    });
    root.appendChild(el);
  });
});
document.addEventListener("play",e=>{document.querySelectorAll("audio").forEach(a=>{if(a!==e.target)a.pause()})},true);
</script></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("build", "serve", "score"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--salt", default="blind", help="build: varies the clip file names")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    return {"build": cmd_build, "serve": cmd_serve, "score": cmd_score}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())

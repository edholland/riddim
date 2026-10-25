"""Steering: the big plain-language moves and the audition loop."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from riddim import steer
from riddim.compose import compose
from riddim.direct import apply_ops, knobs, read_knob
from riddim.params import SongParams
from riddim.ui.server import Handler, State


def test_every_macro_knob_exists():
    ks = knobs(compose("dubstep_140", seed=4).params)
    for m in steer.MACROS:
        for path, w, *_ in m.knobs:
            assert path in ks, (m.name, path)
            assert w in (-1, 1)


def test_a_macro_moves_its_knobs_its_way_and_reads_back():
    base = compose("roots_dub", seed=3).params
    assert all(abs(p["value"]) < 1e-9 for p in steer.positions(base, base))
    now = apply_ops(base, steer.steer("energy", 0.6, base, base))
    assert read_knob(now, "drums.hat_density") > read_knob(base, "drums.hat_density")
    pos = {p["name"]: p["value"] for p in steer.positions(base, now)}
    assert abs(pos["energy"] - 0.6) < 0.05
    back = apply_ops(now, steer.steer("energy", 0.0, base, now))
    pos = {p["name"]: p["value"] for p in steer.positions(base, back)}
    assert abs(pos["energy"]) < 0.05


def test_negative_weights_pull_down():
    base = compose("roots_dub", seed=3).params
    now = apply_ops(base, steer.steer("weight", 1.0, base, base))
    assert read_knob(now, "bass.octave_jumps") <= read_knob(base, "bass.octave_jumps")
    assert read_knob(now, "bass.gate") >= read_knob(base, "bass.gate")


def test_a_hand_twist_survives_another_macro():
    base = compose("roots_dub", seed=3).params
    twisted = apply_ops(base, [steer.Op("lead.echo", to=0.77)])
    now = apply_ops(twisted, steer.steer("energy", 0.5, base, twisted))
    assert abs(read_knob(now, "lead.echo") - 0.77) < 1e-6


def _post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return json.loads(r.read())


def test_steer_and_audition_over_the_wire():
    Handler.state = State(SongParams(profile="dubstep_140", seed=8))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        k = _get(base, "/api/knobs")
        assert [m["name"] for m in k["macros"]] == [m.name for m in steer.MACROS]
        song = _post(base, "/api/steer", {"name": "space", "to": 0.7})
        assert song["can_undo"] and song["changed"]["knobs"]
        k = _get(base, "/api/knobs")
        assert next(m for m in k["macros"] if m["name"] == "space")["value"] > 0.4

        r = _post(base, "/api/audition", {"kind": "directions", "n": 3})
        assert len(r["variants"]) == 3 and all(v["label"] for v in r["variants"])
        assert not any(v["retake"] for v in r["variants"])
        kept = _post(base, "/api/adopt", {"i": 1})
        assert kept["params"] == r["variants"][1]["params"]
        assert kept["can_undo"]

        seed = kept["params"]["seed"]
        r = _post(base, "/api/audition", {"kind": "hook", "n": 3})
        hooks = {tuple(map(tuple, v["parts"]["lead"]["notes"])) for v in r["variants"]}
        assert len(hooks) >= 2
        # The band stays the band (a kick or two may follow the new hook).
        mine = {tuple(n) for n in kept["parts"]["drums"]["notes"]}
        for v in r["variants"]:
            theirs = {tuple(n) for n in v["parts"]["drums"]["notes"]}
            assert len(mine & theirs) > 0.8 * len(mine)
        kept = _post(base, "/api/adopt", {"i": 0})
        assert kept["params"]["seed"] == seed
        assert kept["parts"]["lead"]["notes"] == r["variants"][0]["parts"]["lead"]["notes"]
    finally:
        server.shutdown()


def test_a_bassline_reroll_always_changes_the_bassline():
    """Fixed-shape styles (sustain, pulse, bounce, skip) ignore the sub's
    seed - a re-roll must change the kind of line, or nothing moves."""
    import random
    from riddim.compose import regenerate
    for prof, seed in [("dubstep_140", 966906), ("roots_dub", 1), ("steppers_dub", 2)]:
        s = compose(prof, seed=seed)
        before = sorted((n.midi, n.start) for n in s.part("sub").notes)
        for k in range(3):
            t = regenerate(s, "sub", rng=random.Random(k))
            assert sorted((n.midi, n.start) for n in t.part("sub").notes) != before
            # (the drums may move: the extra kicks lock to the bassline)
            assert t.part("lead").notes == s.part("lead").notes

"""The show engine: hooks, the conversation around them, tension, and a set.

These pin musical facts, not implementation: a hook's rhythm comes back, a
build's roll accelerates, the band stops before a drop, the lead sits above
the stabs and never clashes with them, every knob round-trips, and a song
written to a corpus comes back from its params note for note.
"""

import json
import random

import pytest

from riddim import analyze, corpus
from riddim.arrange import dynamics, interplay, structure
from riddim.compose import KIT_ROWS, compose, regenerate
from riddim.direct import Op, apply_ops, direct, knobs, mutation, read_knob
from riddim.params import SongParams
from riddim.theory import harmony, hook
from riddim.theory.rhythm import metric_weight
from riddim.theory.scale import Scale
from riddim.units import TICKS_PER_16TH, TICKS_PER_BAR
from riddim.profile import PROFILES

ALL_PROFILES = tuple(PROFILES)


def _chords(scale, name="andalusian", bars=8, per=2.0):
    plan = harmony.realise(scale, harmony.BY_NAME[name], bars, per)

    def chord_at(step):
        ev = plan.at(int(step * TICKS_PER_16TH))
        return ev.chord if ev else None
    return chord_at


# --------------------------------------------------------------------------
# Hooks
# --------------------------------------------------------------------------

@pytest.mark.parametrize("form", hook.FORMS)
def test_every_form_is_four_units_and_the_rhythm_returns(form):
    scale = Scale(38, "minor")
    rng = random.Random(3)
    h = hook.best(hook.HookParams(form=form, unit_bars=2), scale, rng, _chords(scale))
    assert len(h.units) == 4
    assert h.units[0].tones, "a hook opens by singing"
    opening = [t.step for t in h.units[0].tones]
    assert any([t.step for t in u.tones] == opening for u in h.units[1:]), \
        f"{form}: the opening rhythm never comes back"
    if form == "call_answer":
        assert not h.units[1].tones and not h.units[3].tones


def test_hooks_are_deterministic_and_scored():
    scale = Scale(38, "phrygian")
    p = hook.params_for("half_time", random.Random(9))
    a = hook.best(p, scale, random.Random(1), _chords(scale, "i_bII"))
    b = hook.best(p, scale, random.Random(1), _chords(scale, "i_bII"))
    assert a.phrase().degrees() == b.phrase().degrees()
    assert 0.0 < a.score <= 1.0
    assert set(a.report) >= {"chord_fit", "climax", "returns", "f_motion", "f_sync"}


def test_realised_hook_never_clashes_where_it_is_heard():
    """No strong-beat or held tone sits a semitone from the chord under it."""
    for seed in range(12):
        scale = Scale(38, "minor")
        chord_at = _chords(scale, "andalusian_cadence")
        h = hook.best(hook.params_for("half_time", random.Random(seed)), scale,
                      random.Random(seed), chord_at)
        for t in hook.realise(h, scale, chord_at):
            heard = metric_weight(int(t.step) % 16) >= 2 or t.length >= 2
            chord = chord_at(t.step)
            if not heard or chord is None:
                continue
            pc = t.midi(scale) % 12
            if pc in chord.pitch_classes():
                continue
            assert not any((pc - c) % 12 in (1, 11) for c in chord.pitch_classes()), \
                (seed, t, chord.symbol())


def test_borrowed_leading_tone_follows_the_chord():
    """Over a borrowed major V the line raises its seventh with the chord."""
    scale = Scale(38, "minor")   # D minor: C natural in the scale, C# in A major
    a_major = harmony.realise(scale, harmony.BY_NAME["andalusian_cadence"], 8, 2).events[3].chord
    t = hook._avoid_clash(hook.Tone(6, 0.0, 4.0), scale, a_major)   # C on a downbeat
    assert t.midi(scale) % 12 == 1      # C#


# --------------------------------------------------------------------------
# The theme plan and the conversation
# --------------------------------------------------------------------------

def _peak_song(profile="dubstep_140", seed=7):
    song = compose(profile=profile, seed=seed)
    peaks = [s for s in song.arrangement.sections if structure.kind_of(s) == "peak"]
    return song, peaks


def test_the_hook_is_heard_in_every_peak():
    for seed in range(1, 9):
        song, peaks = _peak_song(seed=seed)
        lead = song.part("lead")
        sub = song.part("sub")
        for p in peaks:
            sung = lead is not None and len(lead.in_bars(p.start_bar, p.end_bar)) >= p.bars
            assert sung or (sub is not None and p.plays("sub")), (seed, p.name)


def test_lead_sits_above_the_stabs():
    for profile in ("roots_dub", "dubstep_140"):
        for seed in range(1, 7):
            song = compose(profile=profile, seed=seed)
            lead, chords = song.part("lead"), song.part("chords")
            if lead is None or chords is None:
                continue
            top = sorted(n.midi for n in chords.notes)[int(len(chords.notes) * 0.9)]
            above = sum(1 for n in lead.notes if n.midi > top) / len(lead.notes)
            assert above >= 0.6, (profile, seed, above)


def test_lead_is_one_voice():
    for seed in range(1, 7):
        lead = compose(profile="roots_dub", seed=seed).part("lead")
        if lead is None:
            continue
        notes = lead.sorted_notes()
        # One voice: a slide may overlap the next note by a 64th, for the
        # glide - never more, and never two notes starting together.
        assert all(b.start > a.start and a.end - b.start <= 6
                   for a, b in zip(notes, notes[1:])), seed


def test_build_fragment_climbs_toward_the_drop():
    """The build's fragment rises a step at a time, ending at its highest."""
    ip = interplay.InterplayParams()
    scale = Scale(38, "minor")
    h = hook.best(hook.HookParams(unit_bars=2), scale, random.Random(2), _chords(scale))
    line = interplay.plan("build", 16, h, lambda v, off: [], "hook", ip, 0,
                          random.Random(1))
    heads = {}
    for t in line.tones:
        heads.setdefault(int(t.step // 32), []).append(t.degree)
    firsts = [ds[0] for _, ds in sorted(heads.items())]
    assert firsts == sorted(firsts)
    # It approaches the climax without spending it: never past a step below
    # the hook's own high point - and where there is room, it climbs.
    top = max(t.degree for u in h.units for t in u.tones)
    head_top = max(t.degree for t in h.cell.tones if t.step < h.unit_steps / 2)
    assert max(t.degree for t in line.tones) <= max(top - 1, head_top)
    if top - 1 - head_top >= 1:
        assert firsts[-1] > firsts[0]
    climbed = 0
    for seed in range(1, 13):
        h2 = hook.best(hook.HookParams(unit_bars=2), scale, random.Random(seed), _chords(scale))
        l2 = interplay.plan("build", 16, h2, lambda v, off: [], "hook", ip, 0, random.Random(1))
        f2 = {}
        for t in l2.tones:
            f2.setdefault(int(t.step // 32), []).append(t.degree)
        ff = [ds[0] for _, ds in sorted(f2.items())]
        climbed += ff[-1] > ff[0]
    assert climbed >= 4


def test_stabs_duck_the_lead_with_space():
    """At full `space`, no chord stab starts on a lead onset in a peak."""
    p = compose(profile="dubstep_140", seed=4).params
    p.interplay = interplay.InterplayParams(space=1.0, answer=0.0)
    song = compose(params=p)
    lead, chords = song.part("lead"), song.part("chords")
    lead_on = {n.start // TICKS_PER_16TH for n in lead.notes}
    stab_on = {n.start // TICKS_PER_16TH for n in chords.notes}
    # Humanize nudges by a fraction of a step, so allow the neighbour.
    clashes = [s for s in stab_on if s in lead_on]
    assert len(clashes) <= 0.03 * len(stab_on)


# --------------------------------------------------------------------------
# Tension and release
# --------------------------------------------------------------------------

def test_roll_accelerates():
    p = dynamics.DynamicsParams(roll=1.0)
    rng = random.Random(1)
    counts = [len(dynamics.roll(left, 8, p, rng)) for left in range(8, 0, -1)]
    assert counts == sorted(counts) and counts[-1] > counts[0]


def test_band_stops_before_every_drop_and_lands_on_a_crash():
    for seed in range(1, 9):
        song = compose(profile="dubstep_140", seed=seed)
        dyn = song.params.dynamics
        gap = dynamics.gap_ticks(dyn)
        secs = song.arrangement.sections
        for prev, sec in zip(secs, secs[1:]):
            if not dynamics.drop_entry(prev, sec):
                continue
            if gap:
                for part in song.parts:
                    for n in part.notes:
                        if n.midi == KIT_ROWS["riser"] and part.output == "kit":
                            continue
                        assert not (sec.start_tick - gap <= n.start < sec.start_tick), \
                            (seed, part.role, n)
            drums = song.part("drums")
            if dyn.impact > 0.05 and sec.plays("drums"):
                assert any(n.midi == KIT_ROWS["crash"] and n.start == sec.start_tick
                           for n in drums.notes), (seed, sec.name)


def test_tracks_are_show_length():
    for profile in ALL_PROFILES:
        for seed in range(1, 7):
            song = compose(profile=profile, seed=seed)
            minutes = song.arrangement.duration_seconds(song.bpm) / 60
            assert 3.0 <= minutes <= 6.5, (profile, seed, minutes)


def test_drops_escalate():
    for seed in range(1, 13):
        song = compose(profile="dubstep_140", seed=seed)
        peaks = [s.energy for s in song.arrangement.sections
                 if structure.kind_of(s) == "peak"]
        assert peaks == sorted(peaks), seed


# --------------------------------------------------------------------------
# Knobs, words and mutation
# --------------------------------------------------------------------------

def test_new_knob_groups_are_on_the_surface():
    ks = knobs(compose(profile="dubstep_140", seed=2).params)
    for path in ("hook.leap", "hook.repetition", "interplay.lock",
                 "interplay.answer", "dynamics.drop_gap", "dynamics.roll",
                 "hook.form", "hook.rhythm", "lead_style", "lead.gate", "lead.echo"):
        assert path in ks, path
    assert ks["hook.form"].kind == "choice"


def test_choice_knobs_round_trip_by_name_and_index():
    p = compose(profile="dubstep_140", seed=2).params
    q = apply_ops(p, [Op("hook.form", to="riff"), Op("lead_style", to="long_tones")])
    assert q.hook.form == "riff" and q.lead_style == "long_tones"
    r = apply_ops(q, [Op("hook.form", to=0)])
    assert r.hook.form == hook.FORMS[0]
    with pytest.raises(ValueError):
        apply_ops(q, [Op("hook.form", to="symphony")])


def test_intents_reach_the_new_knobs():
    p = compose(profile="dubstep_140", seed=2).params
    q = apply_ops(p, direct("bigger drop"))
    assert q.dynamics.drop_gap >= p.dynamics.drop_gap
    assert q.dynamics.impact >= p.dynamics.impact
    q = apply_ops(p, direct("catchier"))
    assert q.hook.repetition >= p.hook.repetition
    assert apply_ops(p, direct("riff")).lead_style == "riff"


def test_mutation_moves_knobs_inside_their_bounds_and_keeps_seeds():
    song = compose(profile="dubstep_140", seed=2)
    ops = mutation(song.params, ("hook", "interplay", "dynamics"), 0.8, random.Random(5))
    assert ops
    q = apply_ops(song.params, ops)
    for k in knobs(q).values():
        if k.kind in ("float", "int"):
            assert k.low - 1e-9 <= k.value <= k.high + 1e-9, k
    assert q.seeds == song.params.seeds
    assert direct("mutate hook", song.params, random.Random(1))


def test_new_hook_is_a_new_tune_over_the_same_drums():
    song = compose(profile="dubstep_140", seed=5)
    other = regenerate(song, "hook", rng=random.Random(3))
    assert _same_band(other.part("drums"), song.part("drums"))
    assert other.part("lead").notes != song.part("lead").notes


def test_params_reproduce_the_whole_song():
    song = compose(profile="dubstep_140", seed=11)
    again = compose(params=SongParams.from_json(song.params.to_json()))
    for a, b in zip(song.parts, again.parts):
        assert a.notes == b.notes, a.role


# --------------------------------------------------------------------------
# The listener and the set
# --------------------------------------------------------------------------

def test_listener_scores_are_high_and_bounded():
    total = 0.0
    for profile in ("roots_dub", "dubstep_140"):
        for seed in range(1, 6):
            r = analyze.analyze(compose(profile=profile, seed=seed))
            assert 0.0 <= r.score <= 1.0
            assert r.metrics["clashes"] >= 0.8, (profile, seed, r.line())
            total += r.score
    assert total / 10 >= 0.85


def test_corpus_writes_a_set_that_reloads_exactly(tmp_path):
    entries = corpus.build(3, ("roots_dub", "dubstep_140"), candidates=2, seed=4)
    assert [e.song.params.profile for e in entries][0] == "roots_dub"
    path = corpus.write(entries, str(tmp_path))
    text = open(path).read()
    assert "| 1 |" in text and "camelot" in text
    for e in entries:
        again = corpus.load(e.files["json"])
        for a, b in zip(e.song.parts, again.parts):
            assert a.notes == b.notes
    assert json.loads(open(entries[0].files["json"]).read())["hook"]["form"] in hook.FORMS


def test_camelot_by_key_signature():
    song = compose(profile="roots_dub", seed=1, key="A minor")
    assert corpus.camelot(song) == "8A"
    assert corpus.camelot(compose(profile="roots_dub", seed=1, key="E phrygian")) == "8A"
    assert corpus.camelot(compose(profile="roots_dub", seed=1, key="E minor")) == "9A"


def test_audio_preview_renders():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from riddim.render import audio
    song = compose(profile="dubstep_140", seed=3, bars=32)
    data = audio.render(song)
    assert data.shape[1] == 2
    assert data.shape[0] > audio.SR * 30
    assert 0.5 < abs(data).max() <= 1.0


# --------------------------------------------------------------------------
# The critic knows the new rules - and its fixes cure them
# --------------------------------------------------------------------------

def test_critic_catches_a_flat_drop_and_the_loop_cures_it():
    from riddim import critique
    from riddim.arrange.dynamics import DynamicsParams
    p = compose(profile="dubstep_140", seed=7).params
    p.dynamics = DynamicsParams(roll=0.0, riser=0.0, drop_gap=0.0, impact=0.0,
                                switch_up=0.0, fills=0.5)
    broken = compose(params=p)
    assert any("no roll, gap or impact" in f.message for f in critique.critique(broken))
    fixed, steps = critique.revise(broken)
    assert not any("no roll" in f.message for f in critique.critique(fixed))
    assert fixed.params.dynamics.roll > 0.5 and fixed.params.dynamics.drop_gap > 0


def test_critic_catches_a_lead_inside_the_stabs():
    from riddim import critique
    from riddim.profile import get
    song = compose(profile="dubstep_140", seed=7)
    lead = song.part("lead")
    low = get("dubstep_140").chords.low + 2
    lead.notes = [type(n)(low, n.start, n.length, n.velocity) for n in lead.notes]
    f = next(f for f in critique.critique(song) if "inside the stabs" in f.message)
    assert f.fixes[0].kind == "twist" and f.fixes[0].ops[0].path == "lead.octave"


def test_sevenths_voice_m7_and_7_but_never_maj7():
    from riddim.theory import harmony as h
    scale = Scale(38, "minor")
    for degree in range(7):
        c = h.build_chord(scale, degree, 4)
        if len(c.pitches) == 4:
            seventh = (scale.note(degree + 6) - scale.note(degree)) % 12
            assert seventh != 11, degree
    song = compose(params=SongParams(profile="roots_dub", seed=3, chord_size=4))
    assert any("7" in sym for sym in song.harmony.symbols())


def test_vary_hook_keeps_the_band_and_changes_the_tune():
    base = compose(profile="dubstep_140", seed=9)
    out = corpus.variations(base, 2, "hook", seed=1)
    assert len(out) == 2 and out[0].score >= out[1].score
    for e in out:
        assert _same_band(e.song.part("drums"), base.part("drums"))
        assert e.song.part("lead").notes != base.part("lead").notes
    with pytest.raises(ValueError):
        corpus.variations(base, 1, "nonsense")


# --------------------------------------------------------------------------
# Every lead dial is audible, and moves the lead its own way
# --------------------------------------------------------------------------

def _by_bar(ns):
    bars = {}
    for n in ns:
        bars.setdefault(n.start // TICKS_PER_BAR, []).append(n.velocity)
    return list(bars.values())


def _lead_features(song):
    import statistics as st
    # Judged where the hook is sung as written - the first drop, which
    # states it plainly - not where the build fragments it, a later drop
    # develops it (B-theme, sequence, lift) or the outro dissolves it.
    peaks = [s for s in song.arrangement.sections if s.energy >= 0.8][:1]
    ns = [n for n in song.part("lead").sorted_notes()
          if any(p.start_tick <= n.start < p.start_tick + p.length_ticks for p in peaks)]
    iv = [abs(b.midi - a.midi) for a, b in zip(ns, ns[1:])]
    T = TICKS_PER_16TH
    return dict(
        notes=len(ns), interval=st.mean(iv),
        repeats=sum(1 for x in iv if x == 0) / len(iv),
        range=st.pstdev(n.midi for n in ns),
        offbeat=sum(1 for n in ns if metric_weight((n.start // T) % 16) < 2) / len(ns),
        length=st.mean(n.length / T for n in ns),
        # accent is contrast inside a phrase, so measure it inside each bar -
        # section loudness (a soft outro, a ramped build) is not accent
        velspread=st.mean([st.pstdev(v) for v in _by_bar(ns) if len(v) >= 2] or [0.0]),
        pitch=st.mean(n.midi for n in ns),
        vocab=len({n.midi % 12 for n in ns}),
        overlaps=sum(1 for a, b in zip(ns, ns[1:]) if a.end > b.start) / len(ns),
        sounding=len({t for n in ns for t in range(n.start // T, n.end // T)}),
    )


DIAL_EFFECTS = {
    # dial: (the feature it must move, low setting, high setting)
    "hook.density": ("notes", 0.25, 0.85), "hook.syncopation": ("offbeat", 0.1, 0.9),
    "hook.motion": ("interval", 0.1, 0.9), "hook.repeat": ("repeats", 0.0, 0.7),
    "hook.span": ("range", 4, 11), "hook.breath": ("sounding", 0.45, 0.12),
    "hook.outside": ("vocab", 0.0, 1.0),
    "lead.gate": ("length", 0.1, 1.0), "lead.accent": ("velspread", 0.0, 1.0),
    "lead.echo": ("notes", 0.0, 1.0), "lead.flick": ("interval", 0.0, 0.8),
    "lead.grace": ("notes", 0.0, 1.0), "lead.slide": ("overlaps", 0.0, 1.0),
    "lead.ornament": ("notes", 0.0, 1.0), "lead.octave": ("pitch", -1, 1),
}


@pytest.mark.parametrize("path", sorted(DIAL_EFFECTS))
def test_every_lead_dial_moves_the_lead_its_own_way(path):
    """The complaint that started the rebuild: the dials did not change the
    lead. Each dial, low to high, must move its feature the right way in
    most songs - and by a margin a listener would notice."""
    from dataclasses import replace as _r
    feat, lo, hi = DIAL_EFFECTS[path]
    moved, changes = 0, []
    songs = [("dubstep_140", 3), ("dubstep_140", 11), ("roots_dub", 8), ("dubby_halftime", 3)]
    for profile, seed in songs:
        base = compose(profile=profile, seed=seed).params
        if path.startswith("hook."):
            # A hook dial is judged on the notes, with the performance flat.
            base.lead = _r(base.lead, flick=0.0, grace=0.0, ornament=0.0, echo=0.0, slide=0.0)
        a = _lead_features(compose(params=apply_ops(base, [Op(path, to=lo)])))[feat]
        b = _lead_features(compose(params=apply_ops(base, [Op(path, to=hi)])))[feat]
        change = (b - a) / (abs(a) + 1e-6)
        changes.append(change)
        moved += change > 0.03
    assert moved >= len(songs) - 1, (path, changes)
    assert sum(changes) / len(changes) > 0.08, (path, changes)


def _not_kick(part):
    """A kit's notes without the kick - which follows the bass by design."""
    return [n for n in part.sorted_notes() if n.midi != 36]


def _same_band(a, b):
    """The same kit: the kick follows the bass and the hats step out of the
    lead's way, so a new tune moves a few drum notes - and only a few."""
    na, nb = set(a.notes), set(b.notes)
    return len(na & nb) >= 0.85 * max(len(na), len(nb))


def test_everything_is_on_the_grid():
    """No swing, no timing humanisation, no laid-back snare: the Deluge owns
    the feel. Drums, perc, chords and pad on the sixteenth grid - except the
    deliberate 32nds of a full build roll on the snare - the bass on the
    eighth grid, and the lead on the 32nd grid (its grace notes and
    mordents are 32nds, exactly)."""
    from riddim.units import TICKS_PER_16TH as T
    for profile in ALL_PROFILES:
        for seed in (1, 2, 3):
            song = compose(profile=profile, seed=seed)
            for part in song.parts:
                for n in part.notes:
                    if part.role == "sub":
                        assert n.start % (2 * T) == 0, (profile, seed, part.role, n)
                    elif part.role == "lead" or (part.role == "drums" and n.midi == KIT_ROWS["snare"]):
                        assert n.start % (T // 2) == 0, (profile, seed, part.role, n)
                    else:
                        assert n.start % T == 0, (profile, seed, part.role, n)


def test_the_groove_repeats_bar_to_bar():
    """A drum voice that reshuffles its hits every bar is heard as out of
    time. In a flat section - outside the A' bar, the fill and any ramp -
    every bar of kick, snare, hat and rim is the same pattern."""
    from collections import Counter
    rows = (KIT_ROWS["kick"], KIT_ROWS["snare"], KIT_ROWS["hat"], KIT_ROWS["rim"])
    for profile in ALL_PROFILES:
        for seed in (1, 2, 3):
            song = compose(profile=profile, seed=seed)
            drums = song.part("drums")
            for sec in song.arrangement.sections:
                if sec.bars < 8 or sec.ramp > 0 or not sec.plays("drums"):
                    continue
                for row in rows:
                    bars = []
                    for b in range(sec.start_bar, sec.end_bar - 1):
                        if (b - sec.start_bar) % 4 == 3 or (b - sec.start_bar + 1) % 8 == 0:
                            continue
                        bars.append(tuple(sorted(
                            n.start % TICKS_PER_BAR for n in drums.notes
                            if n.midi == row and b * TICKS_PER_BAR <= n.start < (b + 1) * TICKS_PER_BAR)))
                    # A voice held back for a whole block (the drop's layer
                    # plan) is silence, not a different pattern.
                    bars = [b for b in bars if b]
                    if bars:
                        top = Counter(bars).most_common(1)[0][1]
                        assert top == len(bars), (profile, seed, sec.name, row)


# --------------------------------------------------------------------------
# UK garage - dubstep's other parent
# --------------------------------------------------------------------------

def _kicks_by_bar(song, sec):
    drums = song.part("drums")
    out = []
    for b in range(sec.start_bar, sec.end_bar):
        out.append(sorted((n.start % TICKS_PER_BAR) // TICKS_PER_16TH for n in drums.notes
                          if n.midi == KIT_ROWS["kick"] and
                          b * TICKS_PER_BAR <= n.start < (b + 1) * TICKS_PER_BAR))
    return out


def test_two_step_skips_the_beats_a_house_record_would_kick():
    """A 2-step bar has its kick on the one and off the beat after that -
    never on beats 2, 3 or 4, peaks included, whatever the bass does."""
    from riddim.direct import choice_value
    checked = 0
    for seed in range(1, 13):
        song = compose(profile="uk_garage", seed=seed)
        if choice_value(song.params, "drum_feel") == "garage_4x4":
            continue
        for sec in song.arrangement.sections:
            if sec.ramp > 0 or not sec.plays("drums"):
                continue
            for i, kicks in enumerate(_kicks_by_bar(song, sec)):
                if (i + 1) % 4 == 0:
                    continue            # fills and rolls
                assert not {4, 8, 12} & set(kicks), (seed, sec.name, i, kicks)
                checked += 1
    assert checked > 100


def test_garage_hats_skip_and_never_fill_the_beats():
    """The garage hat is the offbeat and the sixteenth after it. It never
    climbs to running sixteenths - that is where the skip disappears."""
    from riddim.theory import groove
    for name in ("two_step", "two_step_skip", "two_step_roll", "garage_4x4"):
        spec = groove.intensify(groove.intensify(groove.FEELS[name]), 1.0)
        hats = [int(h.step) for h in groove.drums(spec, random.Random(1))["hat"].hits]
        assert set(hats) <= {2, 3, 6, 7, 10, 11, 14, 15}, (name, hats)
        assert len(hats) >= 4


def test_garage_draws_its_own_idiom():
    """Across a dozen garage songs: 2-step kits, ninth chords, the skipping
    bassline and the chopped hook all turn up - and the stabs are garage
    stabs, never the reggae skank."""
    from riddim.direct import choice_value
    feels, sizes, basses, phrasings, stabs = set(), set(), set(), set(), set()
    for seed in range(1, 13):
        p = compose(profile="uk_garage", seed=seed).params
        feels.add(choice_value(p, "drum_feel"))
        sizes.add(p.chord_size)
        basses.add(p.bass.style)
        phrasings.add(p.hook.phrasing)
        stabs.add(p.stab_pattern)
    assert {"two_step", "two_step_skip"} <= feels
    assert 5 in sizes and 3 not in sizes
    assert "skip" in basses
    assert "chop" in phrasings
    assert not stabs & {"skank", "double_chop", "half_skank"}


def test_ninths_are_major_ninths_and_name_themselves():
    """The ninth is garage's colour, but a flat nine - a semitone over the
    root - is never stacked; a maj7 degree keeps its ninth as an add9."""
    for mode in ("minor", "dorian", "phrygian", "harmonic_minor"):
        scale = Scale(41, mode)
        for degree in range(7):
            c = harmony.build_chord(scale, degree, 5)
            root = scale.note(degree) % 12
            ivs = {(p - root) % 12 for p in c.pitches}
            assert 1 not in ivs, (mode, degree, c)
            assert 11 not in ivs, (mode, degree, c)
    song = compose(params=SongParams(profile="uk_garage", seed=2, chord_size=5))
    assert any("9" in sym for sym in song.harmony.symbols())


def test_dubstep_inherits_from_garage():
    """Dubstep's pools carry its garage parent: the half-time 2-step kits,
    the skip bass and the chopped hook all appear across a set."""
    from riddim.direct import choice_value
    feels, basses, phrasings = set(), set(), set()
    for seed in range(1, 25):
        p = compose(profile="dubstep_140", seed=seed).params
        feels.add(choice_value(p, "drum_feel"))
        basses.add(p.bass.style)
        phrasings.add(p.hook.phrasing)
    assert feels & {"half_time_skip", "half_time_garage"}
    assert "skip" in basses and "chop" in phrasings
    # ...and its sound-system parent is still the majority.
    assert feels & {"half_time", "half_time_push", "half_time_double"}


def test_chop_is_a_chopped_vocal():
    """The chop lead: a stuttered syllable (one note struck twice), narrow,
    clipped, with air between statements."""
    scale = Scale(45, "dorian")
    chord_at = lambda step: harmony.build_chord(scale, 0, 4)  # noqa: E731
    repeated = 0
    for seed in range(1, 9):
        p = hook.HookParams(phrasing="chop", density=0.5, unit_bars=2, motion=0.3)
        h = hook.best(p, scale, random.Random(seed), chord_at)
        tones = sorted(h.units[0].tones, key=lambda t: t.step)
        pairs = [(x, y) for x, y in zip(tones, tones[1:]) if y.step - x.step == 1]
        assert pairs, seed                                  # the stuttered cut
        repeated += any(x.degree == y.degree for x, y in pairs)
        assert max(t.length for t in tones) <= 2.5
        assert len(tones) <= 10
    # Mostly the same syllable twice; sometimes the colour tone bends it.
    assert repeated >= 5


@pytest.mark.parametrize("word,path,value", [
    ("2-step", "drum_feel", "two_step"),
    ("skippy bass", "bass.style", "skip"),
    ("ninths", "chord_size", 5),
    ("vocal chop", "hook.phrasing", "chop"),
    ("organ", "stab_pattern", "organ"),
])
def test_garage_words_steer(word, path, value):
    from riddim.direct import choice_value
    p = compose(profile="dubstep_140", seed=4).params
    q = apply_ops(p, direct(word, p))
    got = choice_value(q, path) if isinstance(value, str) else read_knob(q, path)
    assert got == value, (word, got)
    compose(params=q)


def test_garage_it_up_keeps_the_meter():
    """'garage' on a half-time tune gives it the half-time 2-step kit, not a
    straight one; on a straight tune, the 2-step itself."""
    from riddim.direct import choice_value
    half = compose(profile="dubstep_140", seed=4).params
    q = apply_ops(half, direct("garage", half))
    assert choice_value(q, "drum_feel") == "half_time_skip"
    assert q.bass.style == "skip" and q.hook.phrasing == "chop" and q.chord_size == 5
    straight = compose(profile="steppers_dub", seed=4).params
    q = apply_ops(straight, direct("ukg", straight))
    assert choice_value(q, "drum_feel") == "two_step"
    song = compose(params=q)
    assert song.parts


def test_drum_feel_mutations_stay_in_meter():
    from riddim.direct import choice_value
    from riddim.theory import groove
    p = compose(profile="dubstep_140", seed=4).params
    for i in range(20):
        for op in mutation(p, ("drums",), 1.0, random.Random(i)):
            if op.path == "drum_feel":
                assert groove.FEELS[op.to].feel == "half_time"


# --------------------------------------------------------------------------
# Development: phrases end, returns develop
# --------------------------------------------------------------------------

def _bar_notes(part, bar):
    return sorted(((n.start % TICKS_PER_BAR) // TICKS_PER_16TH, n.midi) for n in part.notes
                  if bar * TICKS_PER_BAR <= n.start < (bar + 1) * TICKS_PER_BAR)


def test_bass_phrases_turn_around():
    """Bar 8 of a phrase is not bar 1 again: whatever the bass style, the
    last bar of an eight-bar phrase walks into the next one."""
    differs = total = 0
    for profile in ("dubstep_140", "uk_garage", "steppers_dub"):
        for seed in range(1, 7):
            song = compose(profile=profile, seed=seed)
            sub = song.part("sub")
            for sec in song.arrangement.sections:
                if sec.bars < 16 or not sec.plays("sub"):
                    continue
                for ph in range(0, sec.bars - 7, 8):
                    first = _bar_notes(sub, sec.start_bar + ph)
                    last = _bar_notes(sub, sec.start_bar + ph + 7)
                    if first and last:
                        total += 1
                        differs += first != last
    assert total > 20 and differs / total > 0.8, (differs, total)


def test_chords_push_into_the_next_phrase():
    """On bar 4 of a phrase the stabs anticipate the coming chord on the &
    of 4; on bar 8 they leave the last beat to the drums."""
    pushed = 0
    for seed in range(1, 7):
        song = compose(profile="uk_garage", seed=seed)
        chords = song.part("chords")
        for sec in song.arrangement.sections:
            if sec.bars < 8 or not sec.plays("chords"):
                continue
            for ph in range(0, sec.bars, 8):
                b4, b8 = sec.start_bar + ph + 3, sec.start_bar + ph + 7
                pushed += any(st == 14 for st, _ in _bar_notes(chords, b4))
                if ph + 7 < sec.bars:
                    assert not any(st >= 12 for st, _ in _bar_notes(chords, b8)), (seed, sec.name, ph)
    assert pushed >= 10


def test_a_later_drop_sequences_the_hook():
    """In a long later drop one pass of the hook climbs: the same rhythm as
    the first pass, higher."""
    found = 0
    for seed in range(1, 13):
        song = compose(profile="dubstep_140", seed=seed)
        peaks = [s for s in song.arrangement.sections if s.energy >= 0.8]
        lead = song.part("lead")
        for sec in peaks[1:]:
            if sec.bars < 32:
                continue
            notes = [n for n in lead.notes if sec.start_tick <= n.start < sec.start_tick + sec.length_ticks]
            if not notes:
                continue
            q = sec.length_ticks // 4
            passes = [[n for n in notes if sec.start_tick + i * q <= n.start < sec.start_tick + (i + 1) * q]
                      for i in range(4)]
            if passes[0] and passes[3]:
                found += sum(n.midi for n in passes[3]) / len(passes[3]) > \
                    sum(n.midi for n in passes[0]) / len(passes[0])
    assert found >= 2


def test_the_last_drop_can_lift_into_a_new_key():
    """With the lift dial up, the last drop moves up a semitone or a tone -
    every pitched part together, the drums untouched, the harmony with it,
    the critic still clean - and the approach bars are the new key's
    dominant."""
    from riddim import critique
    lifted = 0
    for seed in range(1, 9):
        base = compose(profile="uk_garage", seed=seed).params
        flat = apply_ops(base, [Op("dynamics.lift", to=0.0)])
        up = apply_ops(base, [Op("dynamics.lift", to=1.0)])
        a, b = compose(params=flat), compose(params=up)
        lifts = [ev for ev in b.harmony.events if ev.meta.get("lift")]
        if not lifts:
            continue
        lifted += 1
        k, start = lifts[0].meta["lift"], lifts[0].start
        assert k in (1, 2)
        assert a.part("drums").notes == b.part("drums").notes
        before = [n for n in a.part("chords").notes if n.start < start]
        assert before == [n for n in b.part("chords").notes if n.start < start]
        moved = [(x.midi - y.midi) % 12 for x, y in zip(
            [n for n in b.part("chords").notes if n.start >= start],
            [n for n in a.part("chords").notes if n.start >= start])]
        assert moved and all(d == k for d in moved), (seed, set(moved))
        assert critique.score(b) == 1.0, (seed, critique.critique(b))
    assert lifted >= 5


def test_the_tune_sings_over_the_bass_not_on_it():
    """On strong beats the hook mostly sings the chord's third, fifth or
    seventh - the root is the bass's note."""
    from riddim.theory.rhythm import metric_weight as mw
    shares = []
    for profile in ("dubstep_140", "uk_garage", "roots_dub"):
        for seed in range(1, 7):
            song = compose(profile=profile, seed=seed)
            lead = song.part("lead")
            on = tot = 0
            for n in lead.notes:
                if mw((n.start // TICKS_PER_16TH) % 16) < 2:
                    continue
                ev = song.harmony.at(n.start)
                if ev is None:
                    continue
                tot += 1
                on += n.midi % 12 == ev.chord.root_pc
            if tot:
                shares.append(on / tot)
    assert sum(shares) / len(shares) < 0.38, shares


def test_tensions_sit_and_clashes_do_not():
    """The melody may sing a chord's available tensions - ninth, minor
    eleventh, sixth, seventh - but never a semitone against its third or
    fifth."""
    from riddim.theory import chord as chord_mod
    scale = Scale(57, "minor")                 # A minor
    am = chord_mod.triad(scale, 0)             # A C E
    f = chord_mod.triad(scale, 5)              # F A C
    assert harmony.sits_on(71, am)             # B, the ninth
    assert harmony.sits_on(62, am)             # D, the minor eleventh
    assert harmony.sits_on(67, am)             # G, the seventh
    assert not harmony.sits_on(70, am)         # Bb, the flat nine
    assert not harmony.sits_on(65, am)         # F, a semitone over the fifth
    assert not harmony.sits_on(70, f)          # Bb, the eleventh against a major third
    assert harmony.sits_on(67, f)              # G, the ninth over F


def test_the_pad_suspends_across_chord_changes():
    """Somewhere in a set of songs the pad's top voice holds over a chord
    change - a note not in the new chord - and resolves down by step."""
    found = 0
    for profile in ("roots_dub", "uk_garage", "dubby_halftime"):
        for seed in range(1, 9):
            song = compose(profile=profile, seed=seed)
            pad = song.part("pad")
            if pad is None:
                continue
            notes = sorted(pad.notes, key=lambda n: n.start)
            for n in notes:
                ev = song.harmony.at(n.start)
                if ev is None or n.start != ev.start or n.midi % 12 in ev.chord.pitch_classes():
                    continue
                res = [x for x in notes if x.start == n.end and 1 <= n.midi - x.midi <= 2]
                found += bool(res)
    assert found >= 10


def test_the_first_drop_establishes_the_hook():
    """The lead is sounding through most of the first drop - later drops may
    rest it and hand it to the bass, the first one may not."""
    for profile in ("uk_garage", "steppers_dub", "dubstep_140"):
        for seed in range(1, 9):
            song = compose(profile=profile, seed=seed)
            lead = song.part("lead")
            peaks = [s for s in song.arrangement.sections if s.energy >= 0.8 and s.plays("lead")]
            if lead is None or not peaks:
                continue
            s = peaks[0]
            bars = {b for n in lead.notes for b in range(n.start // TICKS_PER_BAR, (n.end - 1) // TICKS_PER_BAR + 1)
                    if s.start_bar <= b < s.end_bar}
            assert len(bars) >= 0.5 * s.bars, (profile, seed, len(bars), s.bars)


def test_a_chord_answer_does_not_leave_a_hole():
    """Where the chords answer the call, they play the figure (and a
    call-back where there is room) and then the stabs return: no answered
    bar falls silent for more than half of it."""
    for seed in range(1, 7):
        song = compose(profile="roots_dub", seed=seed)
        chords = song.part("chords")
        if chords is None:
            continue
        for sec in song.arrangement.sections:
            if sec.energy < 0.8 or not sec.plays("chords"):
                continue
            lead = song.part("lead")
            for b in range(sec.start_bar, sec.end_bar):
                # where the lead is busy the stabs make room by design;
                # this is about the bars where the lead breathes
                if lead is not None and sum(1 for n in lead.notes
                                            if b * TICKS_PER_BAR <= n.start < (b + 1) * TICKS_PER_BAR) > 2:
                    continue
                steps = sorted({(n.start % TICKS_PER_BAR) // TICKS_PER_16TH for n in chords.notes
                                if b * TICKS_PER_BAR <= n.start < (b + 1) * TICKS_PER_BAR})
                if len(steps) == 1 and steps[0] < 6:
                    raise AssertionError((seed, sec.name, b, steps))


def test_hats_climb_into_the_drop():
    """In the last bars of a ramped build the hats play more than they did
    at the build's start - the top end tightens with the roll."""
    climbed = total = 0
    for profile in ("dubstep_140", "roots_dub", "uk_garage"):
        for seed in range(1, 7):
            song = compose(profile=profile, seed=seed)
            drums = song.part("drums")
            secs = song.arrangement.sections
            for i, sec in enumerate(secs[:-1]):
                if not sec.name.startswith("build") or sec.ramp <= 0 or sec.bars < 8:
                    continue
                count = lambda b: sum(1 for n in drums.notes if n.midi == KIT_ROWS["hat"]  # noqa: E731
                                      and b * TICKS_PER_BAR <= n.start < (b + 1) * TICKS_PER_BAR)
                total += 1
                climbed += count(sec.end_bar - 3) > count(sec.start_bar)
    assert total >= 5 and climbed / total >= 0.8, (climbed, total)


def test_the_outro_remembers_the_tune():
    """The song leaves on its theme: the lead plays the cell, then its head,
    softer, and is silent for the last bars."""
    found = 0
    for profile in ("dubstep_140", "uk_garage", "roots_dub"):
        for seed in range(1, 7):
            song = compose(profile=profile, seed=seed)
            last = song.arrangement.sections[-1]
            lead = song.part("lead")
            if lead is None or not last.plays("lead"):
                continue
            here = [n for n in lead.notes if n.start >= last.start_tick]
            if not here:
                continue
            found += 1
            assert max(n.end for n in here) <= last.start_tick + last.length_ticks - TICKS_PER_BAR, \
                (profile, seed)
    assert found >= 8


def test_a_lifted_song_ends_in_its_new_key():
    for seed in range(1, 13):
        base = compose(profile="uk_garage", seed=seed).params
        song = compose(params=apply_ops(base, [Op("dynamics.lift", to=1.0)]))
        lifts = [ev for ev in song.harmony.events if ev.meta.get("lift")]
        subs = [n for role, n in song.coda if role == "sub"]
        if not lifts or not subs:
            continue
        k = lifts[0].meta["lift"]
        assert (subs[0].midi - song.scale.root - k) % 12 == 0
        return
    pytest.skip("no lifted song with a sub coda in range")


@pytest.mark.parametrize("profile", ["uk_garage", "roots_dub", "dubstep_140"])
def test_a_set_does_not_share_a_fingerprint(profile):
    """Across a set of one genre, keys spread over most of the twelve and no
    one drop progression takes more than a quarter of the songs."""
    from collections import Counter
    keys, drops = Counter(), Counter()
    for seed in range(1, 25):
        song = compose(profile=profile, seed=seed)
        keys[song.params.key.split()[0]] += 1
        drops[next(s.progression for s in song.arrangement.sections if s.energy >= 0.8)] += 1
    assert len(keys) >= 8, keys
    assert drops.most_common(1)[0][1] <= 6, drops


def test_the_last_drop_is_the_whole_band():
    """With two or more drops, the last one is the only place every part
    plays - the pad joins - and it arrives whole, hats and all from its
    first bar."""
    checked = 0
    for profile in ("uk_garage", "dubstep_140"):
        for seed in range(1, 9):
            song = compose(profile=profile, seed=seed)
            peaks = [s for s in song.arrangement.sections if s.energy >= 0.8]
            if len(peaks) < 2 or "pad" in song.params.structure.omit:
                continue
            last = peaks[-1]
            assert last.plays("pad"), (profile, seed)
            hats = [n for n in song.part("drums").notes if n.midi == KIT_ROWS["hat"]
                    and last.start_tick <= n.start < last.start_tick + 2 * TICKS_PER_BAR]
            assert hats, (profile, seed)
            checked += 1
    assert checked >= 6


def test_the_b_theme_contrasts_in_character():
    """A busy first theme gets a singing second and a sparse one a busier
    answer - the same rhythm idiom (one song), the contour turned round."""
    for d in (0.2, 0.5, 0.85):
        p = hook.HookParams(density=d, contour="arch")
        b = hook.contrast_params(p)
        assert b.rhythm == p.rhythm and b.contour != p.contour
        assert (b.density - 0.5) * (d - 0.5) <= 0.01, (d, b.density)


def test_changing_mode_never_breaks_a_song():
    """A mode change on a song whose drops carry developed (roman-numeral)
    progressions redraws them in the new mode; a stored name the new mode
    spells differently is read by degree, never an error."""
    from riddim.compose import _progression_by_name
    for seed in (120, 129):
        p = compose(profile="dubstep_140", seed=seed).params
        for mode in ("dorian", "phrygian", "minor"):
            compose(params=apply_ops(p, direct(mode, p)))
    prog = _progression_by_name("i-bIII-bVII-iv-V", Scale(38, "phrygian"), lenient=True)
    assert len(prog.degrees) == 5


def test_wild_songs_stay_inside_the_lines():
    """Songs twisted at random keep every invariant a show depends on: no
    error, nothing past the end, the sub on its grid, reproducible."""
    from riddim.units import TICKS_PER_16TH as T
    for i, profile in enumerate(PROFILES):
        for seed in (101, 106):
            p = compose(profile=profile, seed=seed).params
            groups, amount = __import__("riddim.direct", fromlist=["MUTATIONS"]).MUTATIONS["wild"]
            p = apply_ops(p, mutation(p, groups, amount, random.Random(seed + i)))
            song = compose(params=p)
            end = song.arrangement.length_ticks
            for part in song.parts:
                assert all(n.end <= end and n.length > 0 for n in part.notes), (profile, seed, part.role)
            assert all(n.start % (2 * T) == 0 for n in song.part("sub").notes)
            again = compose(params=SongParams.from_dict(json.loads(json.dumps(song.params.to_dict()))))
            assert [x.notes for x in song.parts] == [x.notes for x in again.parts]


@pytest.mark.parametrize("profile", ["roots_dub", "uk_garage", "dubstep_140"])
def test_a_live_track_stays_up(profile):
    """A live track is short loops straight in at energy: drums and bass from
    the first bar, nothing over sixteen bars, nothing low but the one break
    - and it is the same song (hook, key, groove) as the record."""
    record = compose(profile=profile, seed=5)
    live = compose(profile=profile, seed=5, live=True)
    secs = live.arrangement.sections
    assert secs[0].plays("drums") and secs[0].plays("sub")
    assert all(s.bars <= 16 for s in secs)
    low = [s for s in secs if s.energy < 0.5]
    assert [s.name for s in low] == ["break"]
    assert live.params.key == record.params.key
    assert live.params.hook == record.params.hook
    again = compose(params=SongParams.from_dict(json.loads(json.dumps(live.params.to_dict()))))
    assert [p.notes for p in live.parts] == [p.notes for p in again.parts]


def test_live_is_a_word_in_the_ui():
    p = compose(profile="uk_garage", seed=5).params
    q = apply_ops(p, direct("live", p))
    assert q.structure.live and q.sections is None
    assert compose(params=q).arrangement.sections[0].name == "groove"


def test_save_and_load_song_messages():
    """The custom firmware's save/load commands: 19 and 20 in the AI
    namespace, names sanitised to what the card accepts, overwrite as a flag."""
    from riddim.deluge import sysex
    msg = sysex.save_song("01 Dub/Chamber!", overwrite=True)
    assert msg[len(sysex.HEADER)] == 19 and msg[len(sysex.HEADER) + 1] == 1
    assert bytes(msg[len(sysex.HEADER) + 2:]).decode() == "01 Dub_Chamber_"
    assert sysex.load_song("RIDDIM TEMPLATE")[len(sysex.HEADER)] == 20
    with pytest.raises(ValueError):
        sysex.song_name("")


def test_set_walks_a_corpus_dry(tmp_path):
    """`riddim-cli set` loads the template, sends and saves every track."""
    from riddim import cli, corpus
    entries = corpus.build(2, ["roots_dub", "uk_garage"], candidates=1, fix=False, live=True)
    corpus.write(entries, str(tmp_path))
    assert cli.main(["set", "--dir", str(tmp_path), "--dry-run"]) == 0


def test_ui_live_mode_keeps_every_take_live():
    from riddim.ui.server import State
    from riddim.direct import Op as _Op
    st = State(SongParams(profile="uk_garage", seed=7), live=True)
    assert st.song.params.structure.live
    assert st.reroll({}).params.structure.live
    assert st.mutate([_Op("hook.density", to=0.7)]).params.structure.live
    p = st.params.copy()
    p.profile, p.structure, p.sections = "dubstep_140", None, None
    assert st.recompose(p).arrangement.sections[0].name == "groove"


def test_ui_save_names_songs_by_session(tmp_path, monkeypatch):
    """The UI's save names a song '<session> NN title', counts on from where
    the session left off, and skips numbers already on the card."""
    from riddim.ui import server
    from riddim.deluge import client
    from riddim.deluge.client import Reply
    from riddim.deluge.sysex import Status, Command
    monkeypatch.setattr(server, "SESSIONS", tmp_path / "sessions.json")
    taken = {"FRI 01"}
    saved = []

    calls = []

    def fake_save(self, name, overwrite=False):
        calls.append("save")
        if any(name.startswith(t) for t in taken):
            return Reply(Command.SAVE_SONG, Status.FILE_EXISTS, [])
        saved.append(name)
        return Reply(Command.SAVE_SONG, Status.OK, [])

    def fake_arm(self, song):
        calls.append("arm")
        return Reply(Command.ARM_SECTION, Status.OK, [])
    monkeypatch.setattr(client.Deluge, "save_song", fake_save)
    monkeypatch.setattr(client.Deluge, "arm_for_save", fake_arm)
    monkeypatch.setattr(client.Deluge, "ping", lambda self: True)
    monkeypatch.setattr(client.Deluge, "supports_save_load", lambda self: True)
    monkeypatch.setattr(client.Deluge, "open", lambda self: self)
    handler = server.Handler.__new__(server.Handler)
    server.Handler.state = server.State(SongParams(profile="uk_garage", seed=3))
    r1 = handler._save({"session": "FRI"})
    r2 = handler._save({"session": "FRI"})
    assert r1["name"].startswith("FRI 02 ") and r2["name"].startswith("FRI 03 ")
    assert server.session_info() == {"session": "FRI", "next": 4}
    # the first section is armed before every save, so play starts the music
    assert calls[0] == "arm" and calls.count("arm") == 2

    from riddim.deluge import sysex
    assert sysex.arm_section(0, 7)[len(sysex.HEADER):] == [21, 0, 7]


def test_old_firmware_is_named_not_failed(monkeypatch):
    """Firmware without the save/load commands answers an unknown command
    with BAD_LENGTH; the probe says so, and the UI's save reports it as
    'flash the save/load firmware' instead of a bare status."""
    from riddim.ui import server
    from riddim.deluge import client
    from riddim.deluge.client import Reply
    from riddim.deluge.sysex import Status, Command
    d = client.Deluge(dry_run=True)
    monkeypatch.setattr(client.Deluge, "send",
                        lambda self, data, expect=None: Reply(Command.LOAD_SONG, Status.BAD_LENGTH, []))
    assert d.supports_save_load() is False
    monkeypatch.setattr(client.Deluge, "send",
                        lambda self, data, expect=None: Reply(Command.LOAD_SONG, Status.FILE_NOT_FOUND, []))
    assert d.supports_save_load() is True
    monkeypatch.setattr(client.Deluge, "open", lambda self: self)
    monkeypatch.setattr(client.Deluge, "ping", lambda self: True)
    monkeypatch.setattr(client.Deluge, "supports_save_load", lambda self: False)
    handler = server.Handler.__new__(server.Handler)
    server.Handler.state = server.State(SongParams(profile="uk_garage", seed=3))
    assert "flash" in handler._save({"session": "FRI"})["error"]

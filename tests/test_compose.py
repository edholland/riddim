"""End-to-end tests: does the composer produce a musically valid song?

The checks here are the ones that would embarrass the tool live - notes out of
key, the bass colliding with the chords, a section that is silent, a song that
changes every time you ask for the same seed.
"""

import pytest

from riddim import profile as profile_mod
from riddim.arrange import dynamics, structure
from riddim.compose import KIT_ROWS, compose
from riddim.song import ALL_DELUGE, Rig
from riddim.units import TICKS_PER_BAR

PROFILES = ["roots_dub", "steppers_dub", "dubstep_140", "dubby_halftime"]


@pytest.fixture(scope="module")
def songs():
    return {name: compose(name, seed=7) for name in PROFILES}


def test_every_profile_composes(songs):
    for name, song in songs.items():
        assert song.note_count() > 200, f"{name} is suspiciously empty"
        assert song.parts, name


def test_deterministic_for_a_seed():
    a = compose("roots_dub", key="G minor", seed=123)
    b = compose("roots_dub", key="G minor", seed=123)
    for pa, pb in zip(a.parts, b.parts):
        assert pa.sorted_notes() == pb.sorted_notes()


def test_different_seeds_give_different_songs():
    a = compose("roots_dub", key="G minor", seed=1)
    b = compose("roots_dub", key="G minor", seed=2)
    assert a.part("sub").sorted_notes() != b.part("sub").sorted_notes()


def test_pitched_parts_stay_in_key(songs):
    """Every pitched note is either a scale tone or a tone of a chord in the
    progression.

    The second case is not a loophole: a progression may deliberately raise a
    third to borrow a leading tone, and that note is correct precisely because
    it is outside the scale. Anything else is a bug.
    """
    for name, song in songs.items():
        allowed = set(song.scale.pitch_classes)
        for event in song.harmony.events:
            allowed |= event.chord.pitch_classes()
        for part in song.parts:
            if part.output == "kit":
                continue
            for note in part.notes:
                assert note.midi % 12 in allowed, (
                    f"{name}/{part.role}: {note} is not in {song.scale} "
                    f"nor in any chord of the progression"
                )


def test_melodic_parts_are_strictly_diatonic(songs):
    """Bass and lead are built from scale degrees. The one way out of the
    scale is to agree with a chord that has already left it - the raised
    leading tone under a borrowed major V - so any chromatic note must be a
    tone of the chord sounding under it."""
    for name, song in songs.items():
        for role in ("sub", "lead"):
            part = song.part(role)
            if part is None:
                continue
            for note in part.notes:
                if song.scale.contains(note.midi):
                    continue
                ev = song.harmony.at(note.start)
                assert ev is not None and ev.chord.contains_pc(note.midi), \
                    f"{name}/{role}: {note} is chromatic and not in the chord"


def test_parts_stay_in_their_register(songs):
    for name, song in songs.items():
        prof = profile_mod.get(name)
        for part in song.parts:
            if part.output == "kit":
                continue
            window = profile_mod.register_for(prof, part.role)
            if part.role == "lead":
                # The lead's window moves with its register dial, with a
                # fifth of headroom for an octave flick.
                from riddim.compose import lead_window
                lo, hi = lead_window(prof, song.params.lead)
                window = profile_mod.Register(lo, hi + 5)
            low, high = part.range()
            assert window.low <= low and high <= window.high, (
                f"{name}/{part.role}: {low}-{high} outside {window}"
            )


def test_sub_never_collides_with_chords(songs):
    """The sub owns the bottom. If the chords reach down into it, the mix is
    mud before it even leaves the Deluge."""
    for name, song in songs.items():
        sub, chords = song.part("sub"), song.part("chords")
        if sub is None or chords is None:
            continue
        assert max(n.midi for n in sub.notes) < min(n.midi for n in chords.notes), name


def test_bass_is_monophonic(songs):
    """One oscillator on the Coral means no overlapping notes, ever."""
    for name, song in songs.items():
        sub = song.part("sub")
        notes = sub.sorted_notes()
        for a, b in zip(notes, notes[1:]):
            assert a.end <= b.start, f"{name}: {a} overlaps {b}"


def test_no_note_escapes_midi_range(songs):
    for song in songs.values():
        for part in song.parts:
            for note in part.notes:
                assert 0 <= note.midi <= 127
                assert 1 <= note.velocity <= 127
                assert note.length >= 1
                assert note.start >= 0


def test_drums_use_valid_kit_rows(songs):
    valid = set(KIT_ROWS.values())
    for song in songs.values():
        for part in song.parts:
            if part.output != "kit":
                continue
            assert {n.midi for n in part.notes} <= valid


def test_every_section_has_something_playing(songs):
    for name, song in songs.items():
        for section in song.arrangement.sections:
            count = sum(
                len(p.in_bars(section.start_bar, section.end_bar)) for p in song.parts
            )
            assert count > 0, f"{name}/{section.name} is silent"


def test_notes_respect_section_mutes(songs):
    """A part must be silent in sections where it is muted - that is the
    entire mechanism the arrangement relies on."""
    for name, song in songs.items():
        for section in song.arrangement.sections:
            for part in song.parts:
                if section.plays(part.role):
                    continue
                assert not part.in_bars(section.start_bar, section.end_bar), (
                    f"{name}: {part.role} plays during muted section {section.name}"
                )


def test_notes_do_not_run_past_the_arrangement(songs):
    for name, song in songs.items():
        limit = song.arrangement.bars * TICKS_PER_BAR
        for part in song.parts:
            for note in part.notes:
                assert note.start < limit, f"{name}/{part.role}"


def test_bass_is_not_the_same_bar_over_and_over(songs):
    """Every bar of the drop used to be byte-identical.

    The turnaround called `ornament`, which only inserts passing notes between
    tones a third apart - and a figure built from repeated roots has no such
    pair, so it silently did nothing. Nothing failed; the bassline was just
    boring. Assert the phrase actually contains more than one bar shape.
    """
    for name, song in songs.items():
        sub = song.part("sub")
        loud = max(
            (s for s in song.arrangement.sections if s.plays("sub")),
            key=lambda s: s.energy,
        )
        shapes = set()
        for bar in range(loud.start_bar, min(loud.end_bar, loud.start_bar + 8)):
            notes = sub.in_bars(bar, bar + 1)
            shapes.add(
                tuple((n.start % TICKS_PER_BAR, n.midi) for n in notes)
            )
        assert len(shapes) > 1, f"{name}: bass repeats one bar verbatim"


def test_half_time_profiles_use_a_half_time_skank():
    """The chord part must not out-note the drums - that was the symptom of
    skanking on the written quarter in a half-time feel."""
    for name in ("dubstep_140", "dubby_halftime"):
        prof = profile_mod.get(name)
        assert prof.beat_steps == 8
        song = compose(name, seed=3)
        chords, drums = song.part("chords"), song.part("drums")
        assert len(chords.notes) < len(drums.notes), name


def test_straight_profiles_keep_the_full_skank():
    for name in ("roots_dub", "steppers_dub"):
        assert profile_mod.get(name).beat_steps == 4


def test_generated_bass_is_never_just_root_and_fifth():
    """A bassline of root and fifth only is harmonically correct and inert.
    The pitch model guarantees a colour tone in any line of three or more
    notes, and the peak - where the bass is the hook - must show it."""
    for seed in range(40):
        for key in (None, "E minor", "G phrygian"):
            song = compose("dubstep_140", key=key, seed=seed)
            sub = song.part("sub")
            if song.params.bass.style != "walk":
                continue   # held, pulsing, bouncing basses sit on the root by design
            peak = max((s for s in song.arrangement.sections if s.plays("sub")),
                       key=lambda s: s.energy)
            in_peak = sub.in_bars(peak.start_bar, peak.start_bar + 4)
            pcs = {n.midi % 12 for n in in_peak}
            frame = {song.scale.root_pc, song.scale.note(4) % 12}
            assert not pcs <= frame, (
                f"seed {seed} key {key}: peak bass is only root and fifth"
            )


def test_hold_figures_are_recognised():
    from riddim.theory import bassline

    assert not bassline.is_melodic(bassline.figure("root_hold"))
    assert not bassline.is_melodic(bassline.figure("sparse_stab"))
    assert not bassline.is_melodic(bassline.figure("half_time_sub"))
    assert bassline.is_melodic(bassline.figure("dotted_push"))
    assert bassline.is_melodic(bassline.figure("phrygian_creep"))


def test_bass_figures_are_not_all_one_note(songs):
    """A figure that only ever plays the root has no melodic content, which is
    fine when a filter is supplying the movement and dull when nothing is."""
    for name, song in songs.items():
        pitches = {n.midi % 12 for n in song.part("sub").notes}
        assert len(pitches) >= 3, f"{name}: bass uses only {pitches}"


def test_energy_drives_density():
    """A high-energy section must be busier than a low-energy one."""
    song = compose("roots_dub", key="A minor", seed=5)
    drums = song.part("drums")
    with_drums = [s for s in song.arrangement.sections if s.plays("drums")]
    quiet = min(with_drums, key=lambda s: s.energy)
    loud = max(with_drums, key=lambda s: s.energy)
    assert loud.energy - quiet.energy > 0.3, "need a real energy gap to test"

    quiet_rate = len(drums.in_bars(quiet.start_bar, quiet.end_bar)) / quiet.bars
    loud_rate = len(drums.in_bars(loud.start_bar, loud.end_bar)) / loud.bars
    assert loud_rate > quiet_rate


def test_automation_only_on_deluge_instruments(songs):
    """Param automation moves a Deluge instrument - a synth, or a kit (firmware
    ai-kit-automation; written only by `send --kits`). A part routed out to
    the Coral has no Deluge filter at all, so a MIDI clip never carries any."""
    for name, song in songs.items():
        for part in song.parts:
            if part.output not in ("synth", "kit"):
                assert not part.automation, (
                    f"{name}/{part.role}: automation on a {part.output} clip "
                    f"would be rejected by the firmware"
                )


def test_default_rig_writes_no_synth_automation():
    """With every pitched part going out to the rack there is no Deluge synth
    to automate, so no filter moves are written for them - all of that is
    played. Only the kits, which stay on the Deluge, carry desk moves."""
    song = compose("roots_dub", seed=7, rig=Rig())
    assert all(p.output in ("midi", "kit") for p in song.parts)
    assert not any(p.automation for p in song.parts if p.output != "kit")
    assert any(p.automation for p in song.parts if p.output == "kit")


def test_all_deluge_rig_does_write_automation():
    """Sketching internally still gets the energy curve as filter automation."""
    song = compose("roots_dub", seed=7, rig=ALL_DELUGE)
    synths = [p for p in song.parts if p.output == "synth"]
    assert synths
    assert any(p.automation for p in synths)


def test_automation_values_are_bounded(songs):
    for song in songs.values():
        for part in song.parts:
            for auto in part.automation:
                assert 0.0 <= auto.start_value <= 1.0
                assert auto.end_value is None or 0.0 <= auto.end_value <= 1.0
                assert auto.length > 0


def test_explicit_key_is_honoured():
    song = compose("roots_dub", key="F# phrygian", seed=3)
    assert song.scale.name == "phrygian"
    assert song.scale.root % 12 == 6


def test_explicit_progression_is_honoured():
    song = compose("dubstep_140", key="E minor", seed=3, progression="andalusian")
    assert song.harmony.progression.name == "andalusian"


def test_unknown_progression_rejected():
    with pytest.raises(ValueError):
        compose("roots_dub", progression="giant_steps")


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        compose("gabber")


def test_bars_argument_scales_the_arrangement():
    short = compose("roots_dub", seed=4, bars=32)
    assert 24 <= short.arrangement.bars <= 44
    assert all(s.bars % 4 == 0 for s in short.arrangement.sections)


def test_all_deluge_rig_keeps_everything_internal():
    song = compose("roots_dub", seed=4, rig=ALL_DELUGE)
    assert all(p.output in ("synth", "kit") for p in song.parts)


def test_default_rig_sends_bass_out_to_the_modular():
    song = compose("roots_dub", seed=4, rig=Rig())
    sub = song.part("sub")
    assert sub.output == "midi" and sub.monophonic


def test_template_override():
    song = compose("roots_dub", seed=4, template="dubstep")
    assert [s.name for s in song.arrangement.sections][:2] == ["intro", "build"]


# --------------------------------------------------------------------------
# Per-section harmony: the song changes its identity where the energy asks
# --------------------------------------------------------------------------

def test_later_drops_return_then_develop():
    """A later drop opens on the first drop's cadence - that is what makes it
    read as a return - and then develops it: the loop again with a chord or
    two bent, so the return is not a repeat."""
    from riddim.compose import _progression_by_name
    developed = 0
    for seed in range(12):
        song = compose("dubstep_140", seed=seed)
        scale = song.scale
        peaks = [_progression_by_name(s.progression, scale).degrees
                 for s in song.arrangement.sections if s.energy >= 0.8]
        first = peaks[0]
        for later in peaks[1:]:
            n = len(first)
            if n > 1:
                assert later[:n] == first, f"seed {seed}: {peaks}"
            assert later[0] == first[0]
            developed += later[n:] != first
    assert developed >= 6


def test_holes_share_a_rest_and_breakdowns_are_the_bridge():
    """A hole's harmony is a place to go, not a destination: intros, dub-outs
    and outros sit on one shared rest chord. A breakdown in the middle of the
    song is its contrast - the bridge - and goes somewhere else."""
    bridges = 0
    for profile in ("dubstep_140", "uk_garage"):
        for seed in range(12):
            song = compose(profile, seed=seed)
            secs = song.arrangement.sections
            rests = [s.progression for i, s in enumerate(secs) if s.energy <= 0.35
                     and not (0 < i < len(secs) - 1 and s.name.startswith("breakdown"))]
            assert len(set(rests)) <= 1, f"seed {seed}: {rests}"
            for i, s in enumerate(secs):
                if 0 < i < len(secs) - 1 and s.name.startswith("breakdown") and s.energy <= 0.35:
                    others = {x.progression for x in secs if x is not s and not x.name.startswith("breakdown")}
                    assert s.progression not in others, (seed, s.name)
                    bridges += 1
    assert bridges >= 5


def test_section_personalities_vary_between_songs():
    names = {
        tuple(s.progression for s in compose("dubstep_140", seed=s).arrangement.sections)
        for s in range(20)
    }
    assert len(names) >= 3, f"only {names}"


def test_explicit_section_progression_is_honoured():
    """A section spec that names its own progression is honoured exactly -
    that is the contract the UI's per-section field rests on."""
    from riddim.params import SongParams

    s0 = compose(params=SongParams(profile="dubstep_140", seed=11))
    p = s0.params.copy()
    i = next(i for i, s in enumerate(s0.arrangement.sections)
             if s.progression != "andalusian" and s.plays("chords"))
    p.sections[i].progression = "andalusian"
    s1 = compose(params=p)
    assert s1.arrangement.sections[i].progression == "andalusian"
    assert s1.params.sections[i].progression == "andalusian"
    # And the section re-voices to it.
    sec = s0.arrangement.sections[i]
    t0, t1 = sec.start_tick, sec.start_tick + sec.length_ticks
    before = [n for n in s0.part("chords").notes if t0 <= n.start < t1]
    after = [n for n in s1.part("chords").notes if t0 <= n.start < t1]
    assert before != after


def test_regenerate_harmony_redraws_the_sections():
    from riddim.compose import regenerate
    from riddim.params import SongParams

    changed = sum(
        [s.progression for s in compose(params=SongParams(profile="dubstep_140", seed=s))
         .arrangement.sections]
        !=
        [s.progression for s in regenerate(
            compose(params=SongParams(profile="dubstep_140", seed=s)), "harmony")
         .arrangement.sections]
        for s in range(6)
    )
    assert changed, "redrawing the harmony never changed a section"
    # The structure itself is untouched - only the personalities moved.
    s0 = compose(params=SongParams(profile="dubstep_140", seed=3))
    s1 = regenerate(s0, "harmony")
    assert [(s.name, s.bars) for s in s0.arrangement.sections] == \
           [(s.name, s.bars) for s in s1.arrangement.sections]


# --------------------------------------------------------------------------
# Ramps: the approach to a peak is audible
# --------------------------------------------------------------------------

def test_bar_energy_math():
    from riddim.arrange import structure

    sec = structure.Section(name="build", bars=8, energy=0.6, ramp=0.5)
    e = [structure.bar_energy(sec, b) for b in range(8)]
    assert e == sorted(e)
    assert e[0] == pytest.approx(0.1)   # 0.6 - 0.5, before the floor
    assert e[-1] == pytest.approx(0.6)  # the peak still gets the full level
    flat = structure.Section(name="riddim", bars=8, energy=0.6, ramp=0.0)
    assert all(structure.bar_energy(flat, b) == 0.6 for b in range(8))
    one = structure.Section(name="x", bars=1, energy=0.6, ramp=0.5)
    assert structure.bar_energy(one, 0) == 0.6


def test_ramped_builds_climb_bar_by_bar():
    """A ramped section plays its own energy curve: the chords get harder
    toward its end instead of landing on a plateau."""
    for seed in range(8):
        song = compose("dubstep_140", seed=seed)
        for sec in song.arrangement.sections:
            if sec.ramp <= 0 or not sec.plays("chords"):
                continue
            chords = song.part("chords")
            if chords is None:
                continue

            def vel(bar):
                t0 = (sec.start_bar + bar) * TICKS_PER_BAR
                ns = [n for n in chords.notes if t0 <= n.start < t0 + TICKS_PER_BAR]
                return sum(n.velocity for n in ns) / len(ns) if ns else None

            first, last = vel(0), vel(sec.bars - 1)
            if first is not None and last is not None:
                assert first < last, f"seed {seed} {sec.name}: {first} -> {last}"


def test_section_ends_with_a_fill_in_the_second_half(songs):
    """The last bar of a section belongs to the fill: snare and rim let go
    of the pattern and run into the next downbeat - never the first half,
    which would pre-empt the phrase the fill is pointing at."""
    for name, song in songs.items():
        drums = song.part("drums")
        if drums is None:
            continue
        secs = song.arrangement.sections
        for i, sec in enumerate(secs):
            if not sec.plays("drums") or sec.bars < 4:
                continue
            # A build into a drop rolls through its last bars instead: the
            # roll is a whole-bar gesture and owns the downbeat.
            if i + 1 < len(secs) and dynamics.drop_entry(sec, secs[i + 1]) \
                    and song.params.dynamics.roll > 0.05:
                continue
            t0 = (sec.start_bar + sec.bars - 1) * TICKS_PER_BAR
            for n in drums.in_bars(sec.start_bar + sec.bars - 1,
                                   sec.start_bar + sec.bars):
                if n.midi in (KIT_ROWS["snare"], KIT_ROWS["rim"]):
                    assert (n.start - t0) >= TICKS_PER_BAR // 2, (
                        f"{name}/{sec.name}: a fill hit in the first half of the bar"
                    )


# --------------------------------------------------------------------------
# Arrangement
# --------------------------------------------------------------------------

def test_sections_tile_the_timeline():
    arrangement = structure.build("dub")
    assert arrangement.sections[0].start_bar == 0
    for a, b in zip(arrangement.sections, arrangement.sections[1:]):
        assert a.end_bar == b.start_bar
    assert arrangement.sections[-1].end_bar == arrangement.bars


def test_at_bar_covers_every_bar():
    arrangement = structure.build("dubstep")
    for bar in range(arrangement.bars):
        assert arrangement.at_bar(bar) is not None
    assert arrangement.at_bar(arrangement.bars) is None


def test_brightness_rises_with_energy():
    values = [structure.brightness_for(e / 10) for e in range(11)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_density_rises_with_energy():
    for role in structure.ROLES:
        values = [structure.density_for(e / 10, role) for e in range(11)]
        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)


def test_unknown_template_rejected():
    with pytest.raises(ValueError):
        structure.build("psytrance")


# --------------------------------------------------------------------------
# Variation between songs
# --------------------------------------------------------------------------

def _drop_kick(song):
    from riddim.units import TICKS_PER_16TH

    drop = max(
        (s for s in song.arrangement.sections if s.plays("drums")),
        key=lambda s: s.energy,
    )
    t0 = drop.start_bar * TICKS_PER_BAR
    grid = ["."] * 16
    for n in song.part("drums").in_bars(drop.start_bar, drop.start_bar + 1):
        if n.midi == KIT_ROWS["kick"]:
            step = round((n.start - t0) / TICKS_PER_16TH)
            if 0 <= step < 16:
                grid[step] = "x"
    return "".join(grid)


@pytest.mark.parametrize("name", PROFILES)
def test_songs_from_one_profile_do_not_all_share_a_skeleton(name):
    """Every song used to have the same section list, kick pattern and stab
    rhythm - a seed only changed the key and the notes, so the tracks all
    sounded like one record. A profile is an idiom to draw from, not a fixed
    skeleton, and this asserts the draws actually differ.
    """
    from collections import Counter

    songs = [compose(name, seed=s) for s in range(30)]

    sequences = {tuple(s.name for s in song.arrangement.sections) for song in songs}
    assert len(sequences) >= 3, f"{name}: only {len(sequences)} structures"

    kicks = {_drop_kick(song) for song in songs}
    assert len(kicks) >= 3, f"{name}: only {len(kicks)} drop kick patterns"

    stabs = {
        (song.params.stab_density, song.params.stab_syncopation) for song in songs
    }
    assert len(stabs) >= 10, f"{name}: only {len(stabs)} stab settings"

    lengths = {song.arrangement.bars for song in songs}
    assert len(lengths) >= 3, f"{name}: every song is the same length"

    progressions = Counter(song.harmony.progression.name for song in songs)
    share = progressions.most_common(1)[0][1] / len(songs)
    assert share <= 0.45, f"{name}: {progressions.most_common(1)} dominates"

    instrumentations = {tuple(p.role for p in song.parts) for song in songs}
    assert len(instrumentations) >= 2, f"{name}: every song has the same parts"


def test_explicit_template_is_exact_and_unmutated():
    """Asking for a template by name is asking for that exact shape."""
    a = compose("dubstep_140", seed=1, template="dubstep")
    b = compose("dubstep_140", seed=2, template="dubstep")
    assert [s.name for s in a.arrangement.sections] == [
        s.name for s in b.arrangement.sections
    ] == ["intro", "build", "drop", "breakdown", "build_2", "drop_2", "outro"]
    assert [s.bars for s in a.arrangement.sections] == [8, 8, 16, 8, 8, 16, 8]


def test_role_omission_removes_empty_sections():
    """A section whose every role was left out is not a section any more."""
    import random

    arr = structure.build("dubstep")
    structure.mutate(arr, random.Random(0), drop_roles=("pad", "lead"), length_ops=0)
    # The dubstep breakdown is pad+lead only, so it must be gone.
    assert "breakdown" not in [s.name for s in arr.sections]
    # And the remaining sections still tile the timeline.
    assert arr.sections[0].start_bar == 0
    for x, y in zip(arr.sections, arr.sections[1:]):
        assert x.end_bar == y.start_bar


def test_mutation_never_shortens_a_drop():
    import random

    for seed in range(50):
        arr = structure.sample("dubstep", random.Random(seed))
        for s in arr.sections:
            if s.energy >= 0.85:
                assert s.bars >= 16, f"seed {seed}: {s.name} is {s.bars} bars"
            assert s.bars % 4 == 0


# --------------------------------------------------------------------------
# Lead presence and bass phrasing
# --------------------------------------------------------------------------

def test_lead_is_sparse_but_present():
    """"Lead is almost never there at all." It was: absent from 35% of songs
    and in 7% of bars of the rest. Sparse is the idiom; absent is a bug."""
    songs = [compose("dubstep_140", seed=s) for s in range(30)]
    with_lead = [s for s in songs if s.part("lead") is not None]
    assert len(with_lead) >= 0.75 * len(songs), "lead left out too often"

    for song in with_lead:
        lead = song.part("lead")
        active = sum(s.bars for s in song.arrangement.sections if s.plays("lead"))
        # Intros and builds only tease the hook; the drops are where it must
        # be heard - in most of their bars.
        peaks = [s for s in song.arrangement.sections
                 if s.plays("lead") and structure.kind_of(s) == "peak"]
        for p in peaks:
            # Bars the lead is *sounding* in - a held landing note sings on.
            bars = len({b for n in lead.in_bars(p.start_bar, p.end_bar)
                        for b in range(n.start // TICKS_PER_BAR,
                                       (n.end - 1) // TICKS_PER_BAR + 1)
                        if b < p.end_bar})
            # `presence` is the dial for how many passes the lead sings (the
            # bass carries the hook through the rest).
            # (Each unit's answer space is the band's, so the lead sings in
            # a little over half the bars of a pass it takes.)
            want = 0.35 * song.params.interplay.presence * p.bars
            assert bars >= want, (
                f"seed {song.seed}: lead in {bars}/{p.bars} bars of {p.name}")
        # A riff, not a wall: well short of a note on every 16th.
        assert len(lead.notes) < 10 * active


@pytest.mark.parametrize("style", ("call_response", "long_tones", "riff"))
def test_every_lead_style_produces_a_line(style):
    from riddim import profile as pm
    from dataclasses import replace

    prof = replace(pm.get("dubstep_140"), lead_styles=(style,), lead_omit_chance=0.0)
    song = compose(prof, seed=9)
    assert style in song.concept
    lead = song.part("lead")
    assert lead is not None and len(lead.notes) > 10
    from riddim.compose import lead_window
    lo, hi = lead_window(prof, song.params.lead)
    for n in lead.notes:
        assert lo <= n.midi <= hi + 5


def test_generated_bass_phrases_restate_then_answer():
    """The cell is restated on non-answer bars and replaced on answer bars,
    so a four-bar phrase is neither a loop nor four unrelated bars."""
    from riddim.params import SongParams
    from riddim.theory.bassline import BassParams

    from riddim.arrange.interplay import InterplayParams
    # The bass's own phrase engine, with the conversation turned off: no
    # replies to the lead, no taking the hook's rhythm.
    params = SongParams(profile="dubstep_140", seed=4, key="G minor",
                        bass=BassParams(density=0.3, syncopation=0.5,
                                        answer_bars=(3,), feel="half_time"),
                        interplay=InterplayParams(presence=1.0, answer=0.0, lock=0.0,
                                                  space=0.0))
    song = compose(params=params)
    sub = song.part("sub")
    peak = max((s for s in song.arrangement.sections if s.plays("sub")),
               key=lambda s: s.energy)
    b = peak.start_bar
    shape = lambda bar: tuple(n.start % TICKS_PER_BAR for n in sub.in_bars(bar, bar + 1))  # noqa: E731
    assert shape(b) == shape(b + 1) == shape(b + 2), "cell not restated"
    assert shape(b + 3) != shape(b), "answer bar identical to the cell"
    notes = sub.sorted_notes()
    for x, y in zip(notes, notes[1:]):
        assert x.end <= y.start


def test_bass_figure_library_has_grown():
    from riddim.theory import bassline

    assert len(bassline.FIGURES) >= 26
    assert sum(bassline.is_melodic(f) for f in bassline.FIGURES) >= 18
    assert any(f.bars == 2 for f in bassline.FIGURES)
    for f in bassline.FIGURES:
        for step, _d, length in f.notes:
            assert 0 <= step < 16 * f.bars and length > 0, f.name


# --------------------------------------------------------------------------
# Params: resolution is complete, regeneration is local
# --------------------------------------------------------------------------

def test_resolved_params_reproduce_the_song_exactly():
    """After one compose, the params describe the song completely: feeding
    them back must give byte-identical notes in every part. This is what
    lets the UI edit one field and trust that nothing else moved."""
    for seed in (1, 77, 300):
        song = compose("dubstep_140", seed=seed)
        again = compose(params=song.params.copy())
        for part in song.parts:
            assert part.sorted_notes() == again.part(part.role).sorted_notes(), (
                f"seed {seed}: {part.role} did not reproduce"
            )


@pytest.mark.parametrize("role", ("drums", "sub", "chords", "lead"))
def test_regenerate_changes_only_that_part(role):
    from riddim.compose import regenerate

    song = compose("dubstep_140", seed=77)
    if song.part(role) is None:
        pytest.skip(f"seed 77 has no {role}")
    new = regenerate(song, role)
    for part in song.parts:
        same = part.sorted_notes() == new.part(part.role).sorted_notes()
        if role == "sub" and part.role == "drums":
            # The kick locks to the bass, so a new bass moves the kick - and
            # only the kick.
            same = [n for n in part.sorted_notes() if n.midi != 36] == \
                [n for n in new.part("drums").sorted_notes() if n.midi != 36]
        if role == "lead" and part.role in ("sub", "chords", "drums"):
            # The band makes room for what the lead plays: a few notes move.
            a, b = set(part.notes), set(new.part(part.role).notes)
            same = len(a & b) >= 0.8 * max(len(a), len(b))
        if part.role == role:
            assert not same, f"{role} was not regenerated"
        else:
            assert same, f"regenerating {role} also changed {part.role}"


def test_recompose_reproduces_the_lead_exactly():
    """The pinned form of the twist model for the lead: after one compose the
    params describe the character of the lead completely, so feeding them back
    must give the same line, note for note. The classic break was the param
    resolution draws shifting the stream the motif was generated from - a
    first compose and a recompose drew different cells."""
    for seed in (1, 3, 12, 77, 300):
        song = compose("dubstep_140", seed=seed)
        lead = song.part("lead")
        if lead is None or not lead.notes:
            continue
        again = compose(params=song.params.copy())
        assert again.part("lead").sorted_notes() == lead.sorted_notes(), (
            f"seed {seed}: the lead did not reproduce")
        # Every other part with it: the whole song is a function of its params.
        for part in song.parts:
            if part.role != "lead":
                assert part.sorted_notes() == again.part(part.role).sorted_notes(), (
                    f"seed {seed}: {part.role} moved on a plain recompose")


def test_regenerate_lead_keeps_character_new_line():
    """A re-roll of the lead is a new *performance* of the same hook: the
    hook (its own seed) and the performance dials stay, the accents, flicks,
    graces and echoes are drawn again. The band makes room for what the lead
    actually plays, so bass, chords and hats may shift a few notes where it
    does - the band stays the band, and pad and perc do not move at all."""
    from riddim.compose import regenerate

    for seed in (1, 5, 9):
        song = compose("dubstep_140", seed=seed)
        if song.part("lead") is None or not song.part("lead").notes:
            continue
        new = regenerate(song, "lead")
        assert new.params.lead == song.params.lead
        assert new.params.hook == song.params.hook
        for part in song.parts:
            if part.role == "lead":
                assert part.sorted_notes() != new.part("lead").sorted_notes(), (
                    f"seed {seed}: the lead was not regenerated")
            elif part.role in ("pad", "perc"):
                assert part.sorted_notes() == new.part(part.role).sorted_notes(), (
                    f"seed {seed}: re-rolling the lead moved {part.role}")
            else:
                a, b = set(part.notes), set(new.part(part.role).notes)
                assert len(a & b) >= 0.8 * max(len(a), len(b)), (
                    f"seed {seed}: re-rolling the lead rewrote {part.role}")


def test_params_round_trip_through_json():
    from riddim.params import SongParams

    song = compose("roots_dub", seed=12)
    text = song.params.to_json()
    back = SongParams.from_json(text)
    again = compose(params=back)
    for part in song.parts:
        assert part.sorted_notes() == again.part(part.role).sorted_notes()


def test_bass_phrase_opens_on_the_root():
    """The downbeat says what the chord is."""
    for seed in range(20):
        song = compose("dubstep_140", seed=seed, key="G minor")
        sub = song.part("sub")
        peak = max((s for s in song.arrangement.sections if s.plays("sub")),
                   key=lambda s: s.energy)
        first = sub.in_bars(peak.start_bar, peak.start_bar + 1)
        if first and first[0].start == peak.start_tick:
            root_pc = song.harmony.at(peak.start_tick).chord.root_pc
            assert first[0].midi % 12 == root_pc, f"seed {seed}"


def test_bass_notes_are_held_not_stabbed():
    """Every bass note used to be exactly one 16th long, whatever the gap
    after it. Durations come from spacing now: a sparse line is long notes.
    Across many seeds, the typical bass note is at least half a beat and a
    good share are a beat or more."""
    import statistics
    from riddim.units import TICKS_PER_16TH

    medians, long_share = [], []
    for seed in range(30):
        song = compose("dubstep_140", seed=seed)
        lens = [n.length / TICKS_PER_16TH for n in song.part("sub").notes]
        medians.append(statistics.median(lens))
        long_share.append(sum(1 for L in lens if L >= 4) / len(lens))
        assert len(set(round(L, 2) for L in lens)) > 3, f"seed {seed}: uniform lengths"
    assert statistics.mean(medians) >= 3.0
    assert statistics.mean(long_share) >= 0.35


def test_bass_gate_controls_hold():
    from riddim.params import SongParams
    from riddim.theory.bassline import BassParams
    from riddim.units import TICKS_PER_16TH
    import statistics

    def median_len(gate):
        from riddim.arrange.interplay import InterplayParams
        # The gate on its own: no yielding to the lead, no hook-locking.
        p = SongParams(profile="dubstep_140", seed=9, key="G minor",
                       bass=BassParams(density=0.15, gate=gate, max_hold=16, feel="half_time"),
                       interplay=InterplayParams(space=0.0, lock=0.0, answer=0.0))
        s = compose(params=p)
        return statistics.mean(n.length / TICKS_PER_16TH for n in s.part("sub").notes)

    assert median_len(0.95) > median_len(0.5) * 1.4


def test_the_desk_pulls_the_bass_in_dub():
    """Dub's oldest desk move: the sub muted for a bar on a phrase's turn
    and slammed back on the next downbeat - never in a section's first bar,
    and only where the dynamics ask for it."""
    import random as _r
    from riddim.arrange import dynamics
    from riddim.arrange.structure import Section
    sec = Section(name="peak", bars=32, energy=0.9, start_bar=8)
    wins = dynamics.bass_drop_windows(sec, dynamics.DynamicsParams(bass_drops=2.0), _r.Random(1))
    assert wins
    for a, b in wins:
        bar = a // TICKS_PER_BAR - sec.start_bar
        assert bar > 0 and (b // TICKS_PER_BAR - sec.start_bar - 1) % 8 in (3, 7)
        assert 1 <= (b - a) // TICKS_PER_BAR <= 2
    assert not dynamics.bass_drop_windows(sec, dynamics.DynamicsParams(bass_drops=0.0), _r.Random(1))
    # The dub family uses it: somewhere in a set the sub drops out mid-section.
    dropped = 0
    for seed in range(1, 9):
        song = compose("roots_dub", seed=seed)
        dropped += song.params.dynamics.bass_drops > 0
    assert dropped == 8


def test_dub_keys_bubble():
    """A roots or steppers record's pad track is the organ bubble: short
    chords on each beat's 'e' and '&', the root left to the bass - not a
    held string pad. The other families keep their held pad."""
    from riddim.units import TICKS_PER_16TH
    for name in ("roots_dub", "steppers_dub"):
        assert profile_mod.get(name).pad_style == "bubble"
        song = compose(name, seed=5)
        notes = song.part("pad").notes
        assert notes
        for n in notes:
            assert n.length <= TICKS_PER_16TH
            assert (n.start // TICKS_PER_16TH) % 4 in (1, 2)
    held = compose("dubstep_140", seed=5).part("pad").notes
    assert held and max(n.length for n in held) > TICKS_PER_BAR // 2

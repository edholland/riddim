"""Tests for the theory core.

These check musical facts, not just that the code runs. If a test here fails,
something is genuinely out of key or out of time.
"""

import random

import pytest

from riddim.theory import bassline, chord, harmony, melody, pitch, rhythm, voicing
from riddim.theory.scale import Scale, parse_key


# --------------------------------------------------------------------------
# Pitch
# --------------------------------------------------------------------------

def test_note_names_round_trip():
    for midi in range(24, 108):
        assert pitch.parse_note(pitch.note_name(midi)) == midi


def test_deluge_octave_convention():
    # The Deluge displays MIDI 60 as C3.
    assert pitch.note_name(60) == "C3"
    assert pitch.parse_note("C3") == 60
    assert pitch.parse_note("A1") == 45


def test_parse_note_rejects_garbage():
    for bad in ("", "H3", "C", "C#x"):
        with pytest.raises(ValueError):
            pitch.parse_note(bad)


def test_transpose_into_window():
    # A note far above the window folds down by whole octaves, keeping its
    # pitch class.
    placed = pitch.transpose_into(84, 36, 48)
    assert 36 <= placed <= 48
    assert placed % 12 == 84 % 12


# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------

def test_minor_scale_intervals():
    scale = Scale(pitch.parse_note("A2"), "minor")
    names = [pitch.note_name(scale.note(d)) for d in range(7)]
    assert names == ["A2", "B2", "C3", "D3", "E3", "F3", "G3"]


def test_degree_wraps_octaves():
    scale = Scale(60, "minor")
    assert scale.note(7) == 72       # tonic an octave up
    assert scale.note(-1) == 58      # b7 below the tonic
    assert scale.note(14) == 84


def test_degree_of_is_inverse_of_note():
    scale = Scale(45, "dorian")
    for degree in range(-7, 15):
        note = scale.note(degree)
        assert scale.degree_of(note) == degree


def test_degree_of_returns_none_off_scale():
    scale = Scale(45, "minor")          # A minor has no A#
    assert scale.degree_of(46) is None


def test_snap_lands_in_scale():
    scale = Scale(45, "phrygian")
    for midi in range(40, 80):
        assert scale.contains(scale.snap(midi))


def test_phrygian_has_flat_second():
    scale = Scale(60, "phrygian")
    assert scale.note(1) - scale.note(0) == 1


def test_parse_key_defaults_to_bass_octave():
    scale = parse_key("G minor")
    assert 36 <= scale.root <= 47
    assert scale.name == "minor"
    assert parse_key("F#1 phrygian").name == "phrygian"


def test_unknown_scale_rejected():
    with pytest.raises(ValueError):
        Scale(60, "hungarian_minor_but_spicier")


# --------------------------------------------------------------------------
# Chords
# --------------------------------------------------------------------------

def test_diatonic_triads_in_minor():
    scale = Scale(pitch.parse_note("A2"), "minor")
    assert chord.triad(scale, 0).quality == "m"    # i
    assert chord.triad(scale, 2).quality == ""     # bIII major
    assert chord.triad(scale, 6).quality == ""     # bVII major
    assert chord.triad(scale, 1).quality == "dim"  # ii diminished


def test_seventh_chords():
    scale = Scale(pitch.parse_note("A2"), "minor")
    assert chord.seventh(scale, 0).quality == "m7"
    assert chord.seventh(scale, 6).quality == "7"   # bVII7 is dominant


def test_all_chord_tones_are_in_scale():
    scale = Scale(43, "dorian")
    for degree in range(7):
        for note in chord.seventh(scale, degree).pitches:
            assert scale.contains(note)


def test_quartal_voicing_stacks_fourths():
    scale = Scale(60, "minor")
    q = chord.quartal(scale, 0, size=3)
    gaps = [b - a for a, b in zip(q.pitches, q.pitches[1:])]
    assert all(4 <= g <= 6 for g in gaps)  # perfect or augmented fourths


def test_inversions_preserve_pitch_classes():
    scale = Scale(60, "minor")
    source = chord.triad(scale, 0)
    for inv in chord.inversions(source):
        assert inv.pitch_classes() == source.pitch_classes()


def test_voicings_stay_in_register():
    scale = Scale(48, "minor")
    for candidate in chord.voicings(chord.triad(scale, 0), 55, 79):
        assert min(candidate.pitches) >= 55
        assert max(candidate.pitches) <= 79


def test_rootless_drops_the_bottom_note():
    scale = Scale(60, "minor")
    source = chord.seventh(scale, 0)
    assert len(chord.rootless(source).pitches) == len(source.pitches) - 1


# --------------------------------------------------------------------------
# Voice leading
# --------------------------------------------------------------------------

def test_voice_leading_beats_root_position():
    """The whole point: led voicings move less than naive root position."""
    scale = Scale(45, "minor")
    progression = [chord.triad(scale, d) for d in (0, 5, 6, 3)]

    led = voicing.lead(progression, 55, 79)
    led_movement = sum(
        voicing.voicing_distance(a.pitches, b.pitches) for a, b in zip(led, led[1:])
    )

    naive = [c.transposed(12 * ((55 - min(c.pitches)) // 12 + 1)) for c in progression]
    naive_movement = sum(
        voicing.voicing_distance(a.pitches, b.pitches)
        for a, b in zip(naive, naive[1:])
    )

    assert led_movement < naive_movement


def test_voice_leading_keeps_the_chords():
    scale = Scale(45, "minor")
    progression = [chord.triad(scale, d) for d in (0, 6, 5, 6)]
    for source, led in zip(progression, voicing.lead(progression, 55, 79)):
        assert led.pitch_classes() == source.pitch_classes()


def test_voice_leading_respects_register():
    scale = Scale(45, "minor")
    progression = [chord.triad(scale, d) for d in range(7)]
    for led in voicing.lead(progression, 60, 72):
        assert min(led.pitches) >= 60 and max(led.pitches) <= 72


# --------------------------------------------------------------------------
# Rhythm
# --------------------------------------------------------------------------

def test_euclid_produces_tresillo():
    assert rhythm.bjorklund(3, 8) == [
        True, False, False, True, False, False, True, False
    ]


def test_euclid_produces_cinquillo():
    assert rhythm.bjorklund(5, 8) == [
        True, False, True, True, False, True, True, False
    ]


def test_euclid_degenerate_cases():
    assert rhythm.bjorklund(0, 8) == [False] * 8
    assert rhythm.bjorklund(8, 8) == [True] * 8
    assert rhythm.bjorklund(12, 8) == [True] * 8
    assert rhythm.bjorklund(3, 0) == []


def test_euclid_pulse_count_is_exact():
    for steps in (8, 12, 16):
        for pulses in range(steps + 1):
            assert sum(rhythm.bjorklund(pulses, steps)) == pulses


def test_euclid_is_maximally_even():
    """Gaps between onsets differ by at most one step - that is the defining
    property of a Euclidean rhythm."""
    for steps in (8, 12, 16):
        for pulses in range(2, steps):
            flags = rhythm.bjorklund(pulses, steps)
            onsets = [i for i, f in enumerate(flags) if f]
            gaps = [b - a for a, b in zip(onsets, onsets[1:])]
            gaps.append(steps - onsets[-1] + onsets[0])
            assert max(gaps) - min(gaps) <= 1


def test_metric_weight_hierarchy():
    assert rhythm.metric_weight(0) == 4.0     # bar
    assert rhythm.metric_weight(8) == 3.0     # half bar
    assert rhythm.metric_weight(4) == 2.0     # beat
    assert rhythm.metric_weight(2) == 1.0     # eighth
    assert rhythm.metric_weight(1) == 0.0     # sixteenth


def test_skank_is_all_offbeats():
    pattern = rhythm.skank()
    assert pattern.steps() == [2, 6, 10, 14]
    assert all(rhythm.metric_weight(s) < 2.0 for s in pattern.steps())


def test_half_time_skank_lands_on_the_felt_offbeats():
    """In half-time the felt beat is a half note, so the skank is two stabs a
    bar - on written beats 2 and 4 - not four. Using the written quarter turns
    a skank into double-time chatter."""
    straight = rhythm.skank()
    half = rhythm.skank(beat_steps=8)
    assert straight.steps() == [2, 6, 10, 14]
    assert half.steps() == [4, 12]


def test_is_offbeat_respects_the_felt_beat():
    assert [s for s in range(16) if rhythm.is_offbeat(s, 16, 8)] == [4, 12]
    assert [s for s in range(16) if rhythm.is_offbeat(s, 16, 4)] == [2, 6, 10, 14]


def test_one_drop_leaves_beat_one_empty():
    """The whole identity of one-drop is the missing downbeat."""
    kick = rhythm.riddim("one_drop", "kick")
    assert 0 not in kick.steps()
    assert 8 in kick.steps()


def test_half_time_snare_on_beat_three():
    snare = rhythm.riddim("half_time", "snare")
    assert snare.steps() == [8]


def test_thin_keeps_the_strong_beats():
    pattern = rhythm.from_string("xxxxxxxxxxxxxxxx")
    thinned = rhythm.thin(pattern, 0.25, random.Random(0))
    assert len(thinned) == 4
    assert 0 in thinned.steps()  # the downbeat always survives


def test_thin_is_monotonic():
    pattern = rhythm.from_string("x.x.x.x.x.x.x.x.")
    rng = random.Random(1)
    sizes = [len(rhythm.thin(pattern, k / 10, rng)) for k in range(1, 11)]
    assert sizes == sorted(sizes)


def test_thicken_adds_without_collisions():
    pattern = rhythm.from_string("x...x...x...x...")
    thick = rhythm.thicken(pattern, 4, random.Random(0))
    steps = thick.steps()
    assert len(steps) == len(set(steps)) == 8


def test_swing_delays_only_odd_steps():
    pattern = rhythm.from_string("xxxx")
    swung = rhythm.swing(pattern, 0.25)
    assert [h.step for h in swung.sorted_hits()] == [0.0, 1.25, 2.0, 3.25]


def test_rotate_wraps_within_length():
    pattern = rhythm.from_string("x..x..x.")
    assert sorted(rhythm.rotate(pattern, 2).steps()) == [0, 2, 5]


def test_clave_patterns_are_16_steps():
    for name in rhythm.CLAVES:
        assert rhythm.clave(name).length == 16


def test_unknown_riddim_rejected():
    with pytest.raises(ValueError):
        rhythm.riddim("jungle", "kick")
    with pytest.raises(ValueError):
        rhythm.riddim("one_drop", "tuba")


# --------------------------------------------------------------------------
# Melody
# --------------------------------------------------------------------------

def test_generated_melody_is_all_scale_tones():
    scale = Scale(60, "phrygian")
    motif = melody.generate(rhythm.euclid(5, 16), rng=random.Random(3))
    for midi, _, _, _ in motif.notes(scale):
        assert scale.contains(midi)


def test_melody_is_mostly_stepwise():
    """Melodies that leap constantly sound like arpeggiator output."""
    scale = Scale(60, "minor")
    motif = melody.generate(
        rhythm.euclid(7, 16), contour="arch", rng=random.Random(7)
    )
    degrees = motif.degrees()
    moves = [abs(b - a) for a, b in zip(degrees, degrees[1:])]
    stepwise = sum(1 for m in moves if m <= 1)
    assert stepwise / len(moves) >= 0.6


def test_melody_lands_chord_tones_on_strong_beats():
    scale = Scale(60, "minor")
    target = chord.triad(scale, 0)
    motif = melody.generate(
        rhythm.from_string("x...x...x...x..x"),
        chord=target, scale=scale, rng=random.Random(11),
    )
    strong = [
        t for t in motif.tones if rhythm.metric_weight(int(t.step)) >= 2.0
    ]
    assert strong
    for tone in strong:
        assert scale.note(tone.degree) % 12 in target.pitch_classes()


def test_transposition_stays_diatonic():
    scale = Scale(60, "dorian")
    motif = melody.generate(rhythm.euclid(5, 16), rng=random.Random(5))
    for midi, _, _, _ in melody.transpose(motif, 3).notes(scale):
        assert scale.contains(midi)


def test_inversion_mirrors_contour():
    scale = Scale(60, "minor")
    motif = melody.generate(
        rhythm.from_string("x.x.x.x."), contour="ascending", rng=random.Random(2)
    )
    inverted = melody.invert(motif)
    for a, b in zip(motif.sorted_tones(), inverted.sorted_tones()):
        assert a.degree + b.degree == 2 * motif.sorted_tones()[0].degree


def test_inversion_is_an_involution():
    motif = melody.generate(rhythm.euclid(5, 16), rng=random.Random(9))
    twice = melody.invert(melody.invert(motif))
    assert twice.degrees() == motif.degrees()


def test_retrograde_reverses_pitches_but_keeps_rhythm():
    motif = melody.generate(rhythm.euclid(5, 16), rng=random.Random(4))
    back = melody.retrograde(motif)
    assert back.degrees() == motif.degrees()[::-1]
    assert [t.step for t in back.sorted_tones()] == [
        t.step for t in motif.sorted_tones()
    ]


def test_augment_doubles_time():
    motif = melody.generate(rhythm.euclid(3, 16), rng=random.Random(6))
    stretched = melody.augment(motif, 2.0)
    assert stretched.length == motif.length * 2
    for a, b in zip(motif.sorted_tones(), stretched.sorted_tones()):
        assert b.step == a.step * 2
        assert b.length == a.length * 2


def test_displace_wraps_inside_the_bar():
    motif = melody.generate(rhythm.euclid(5, 16), rng=random.Random(8))
    for tone in melody.displace(motif, 3).tones:
        assert 0 <= tone.step < motif.length


def test_sequence_repeats_at_transpositions():
    motif = melody.generate(rhythm.from_string("x.x.x.x."), rng=random.Random(1))
    seq = melody.sequence(motif, [0, -1, -2])
    assert len(seq.tones) == len(motif.tones) * 3
    assert seq.length == motif.length * 3


def test_develop_stages_are_distinct():
    motif = melody.generate(rhythm.euclid(5, 16), rng=random.Random(13))
    names = {melody.develop(motif, s, random.Random(13))[1] for s in range(9)}
    assert len(names) == 9


def test_contour_sampling_is_bounded():
    for name in melody.CONTOURS:
        for i in range(11):
            assert 0.0 <= melody.sample_contour(name, i / 10) <= 1.0


# --------------------------------------------------------------------------
# Harmony
# --------------------------------------------------------------------------

def test_every_progression_has_a_valid_mode():
    from riddim.theory.scale import SCALES

    for prog in harmony.PROGRESSIONS:
        assert prog.modes
        for mode in prog.modes:
            assert mode in SCALES


def test_no_progression_rests_on_a_diminished_chord():
    """Every chord of every progression must be stable in every mode the
    progression claims to support.

    A diminished or augmented triad cannot hold down a bar of a vamp - it is
    a passing chord. This caught three real mistakes: the Andalusian cadence
    in phrygian and in harmonic minor, and i-v in phrygian, all of which
    produced a diminished chord exactly where the progression needed to rest.
    """
    offenders = []
    for prog in harmony.PROGRESSIONS:
        for mode in prog.modes:
            scale = Scale(45, mode)
            for degree in prog.degrees:
                built = chord.triad(scale, degree)
                if degree in prog.major_degrees:
                    built = chord.with_major_third(built)
                if built.quality in ("dim", "aug", "?"):
                    offenders.append(f"{prog.name}/{mode}: {built.symbol()}")
    assert not offenders, offenders


def test_choose_never_returns_an_unsafe_progression():
    """`choose` must be safe for every mode any profile can pick, including
    through its fallback path.

    The earlier version of this test only checked each progression against its
    *declared* modes, which missed the real bug: removing harmonic_minor from
    the last progression that declared it left `for_mode` empty, and the
    fallback then handed a phrygian-only vamp to a harmonic-minor key,
    producing a diminished chord as the second chord of the tune.
    """
    from riddim import profile as profile_mod

    modes = {m for p in profile_mod.PROFILES.values() for m in p.modes}
    assert modes, "no profile declares any mode"

    rng = random.Random(0)
    for mode in sorted(modes):
        for tension in (0.0, 0.25, 0.5, 0.75, 1.0):
            for _ in range(20):
                prog = harmony.choose(mode, tension, rng)
                assert harmony.is_safe_in(prog, mode), (
                    f"{prog.name} is not safe in {mode}"
                )


def test_every_profile_mode_has_a_preferred_progression():
    """Not just a safe fallback - each mode should have progressions that
    actually declare it, or the library is leaning on the fallback."""
    from riddim import profile as profile_mod

    for prof in profile_mod.PROFILES.values():
        for mode in prof.modes:
            assert harmony.for_mode(mode), f"{prof.name}: nothing declares {mode}"


def test_is_safe_in_catches_diminished():
    # i-v in phrygian gives a diminished v - the bug this guards against.
    assert not harmony.is_safe_in(harmony.BY_NAME["i_v"], "phrygian")
    assert harmony.is_safe_in(harmony.BY_NAME["i_v"], "minor")


def test_major_third_alteration_makes_a_major_v():
    scale = Scale(45, "minor")                     # A minor
    five = chord.triad(scale, 4)                   # Em
    assert five.quality == "m"
    raised = chord.with_major_third(five)
    assert raised.quality == ""                    # E major
    assert raised.root_pc == five.root_pc
    # The leading tone it introduces is deliberately outside the scale.
    assert any(not scale.contains(p) for p in raised.pitches)


def test_andalusian_cadence_resolves():
    scale = Scale(45, "minor")
    plan = harmony.realise(scale, harmony.BY_NAME["andalusian_cadence"], bars=8,
                           bars_per_chord=2)
    assert plan.symbols() == ["Am", "G", "F", "E"]


def test_realise_covers_the_whole_arrangement():
    scale = Scale(45, "minor")
    plan = harmony.realise(scale, harmony.BY_NAME["i_bVI_bVII"], bars=16,
                           bars_per_chord=2)
    assert plan.events[0].start == 0
    assert plan.events[-1].end == 16 * 384
    for a, b in zip(plan.events, plan.events[1:]):
        assert a.end == b.start  # no gaps


def test_at_finds_the_sounding_chord():
    scale = Scale(45, "minor")
    plan = harmony.realise(scale, harmony.BY_NAME["i_bVII"], bars=8, bars_per_chord=2)
    assert plan.at(0) is plan.events[0]
    assert plan.at(384 * 2) is plan.events[1]
    assert plan.at(384 * 8) is None


def test_voiced_is_rootless_and_in_register():
    scale = Scale(45, "minor")
    plan = harmony.realise(scale, harmony.BY_NAME["andalusian"], bars=8)
    voiced = harmony.voiced(plan, 55, 79)
    for event in voiced.events:
        assert min(event.chord.pitches) >= 55
        assert max(event.chord.pitches) <= 79


def test_choose_respects_mode():
    rng = random.Random(0)
    for _ in range(20):
        assert "phrygian" in harmony.choose("phrygian", 0.7, rng).modes


def test_tonic_drone_is_one_chord():
    assert harmony.BY_NAME["tonic_drone"].degrees == (0,)


# --------------------------------------------------------------------------
# Bassline
# --------------------------------------------------------------------------

def test_every_figure_fits_its_bars():
    for fig in bassline.FIGURES:
        for step, _degree, length in fig.notes:
            assert 0 <= step < 16 * fig.bars
            assert length > 0


def test_figure_realises_to_scale_tones():
    scale = Scale(45, "minor")
    for fig in bassline.FIGURES:
        motif = bassline.from_figure(fig, root_degree=0)
        for midi, _, _, _ in motif.notes(scale):
            assert scale.contains(midi)


def test_bass_stays_in_the_sub_register():
    scale = Scale(45, "minor")
    motif = bassline.from_figure(bassline.figure("octave_drop"))
    for midi, _, _, _ in bassline.constrain_register(motif, scale, 28, 48):
        assert 28 <= midi <= 48


def test_monophonic_removes_all_overlap():
    notes = [(36, 0.0, 8.0, 100), (38, 2.0, 8.0, 100), (40, 3.0, 2.0, 100)]
    mono = bassline.monophonic(notes)
    for (_, start, length, _), (_, next_start, _, _) in zip(mono, mono[1:]):
        assert start + length <= next_start + 1e-9


def test_monophonic_preserves_note_count():
    notes = [(36, float(i), 4.0, 100) for i in range(8)]
    assert len(bassline.monophonic(notes)) == len(notes)


def test_generated_bass_favours_root_on_strong_beats():
    pattern = rhythm.from_string("x...x...x...x...")
    motif = bassline.generate(pattern, root_degree=0, rng=random.Random(0))
    downbeat = [t for t in motif.tones if t.step == 0.0][0]
    assert downbeat.degree == 0


def test_enumerated_progressions_are_complete_safe_and_named():
    """The menu the UI shows must be every cycle the grammar allows in the
    mode - all safe, all uniquely named, sorted restful to tense, and every
    name must parse back into the same degrees."""
    from riddim.compose import _parse_roman

    for mode in ("minor", "phrygian", "dorian", "harmonic_minor"):
        scale = Scale(45, mode)
        items = harmony.enumerate_progressions(scale)
        assert len(items) >= 15, mode
        names = [p.name for p in items]
        assert len(names) == len(set(names)), f"{mode}: duplicate names"
        assert names[0] == "i"
        tensions = [p.tension for p in items]
        assert tensions == sorted(tensions)
        for p in items:
            assert harmony.is_safe_in(p, mode), f"{mode}: {p.name} unsafe"
            back = _parse_roman(p.name, scale)
            assert back.degrees == p.degrees and back.major_degrees == p.major_degrees, (
                f"{mode}: {p.name} did not round-trip"
            )

"""Compose a complete song from parameters.

The pipeline:

1. **Resolve** the params. Anything left unset is drawn from the song seed -
   key, structure, drum spec, bass params, harmony tension - and written
   back, so the returned song's params describe it completely.
2. **Generate** the material from those params: the arrangement from the
   energy grammar, the progression from the root-motion grammar, the drum
   feel from a groove spec, the bass phrase from its pitch model, the motif
   from the contour generator.
3. **Realise** each part, section by section, from its own random stream.

Nothing in step 2 is a lookup. The named figures, riddims and templates in
the library still exist and can be asked for by name, but the default path
never touches them.

Each part is generated from `params.seed_for(role)`, so regenerating one
part is a new seed for that part and nothing else moves.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from . import profile as profile_mod
from .arrange import structure
from .arrange.structure import Arrangement, Section, StructureParams
from .params import PARTS, SectionSpec, SongParams
from .profile import Profile
from .song import Automation, Note, Part, Rig, Song
from .theory import (
    bassline, chord as chord_mod, fill, groove, harmony, melody, pitch, rhythm,
)
from .theory.scale import Scale, parse_key
from .units import STEPS_PER_BAR, TICKS_PER_16TH, TICKS_PER_BAR

# Deluge kit rows: note 36 is the bottom visible row, one semitone per row up.
KIT_ROWS = {"kick": 36, "snare": 37, "hat": 38, "rim": 39, "perc": 40, "shaker": 41}


@dataclass
class Material:
    """Everything generated once per song that the part builders read."""

    scale: Scale
    plan: harmony.HarmonyPlan
    voiced: harmony.HarmonyPlan
    motif: melody.Motif
    drums: groove.DrumSpec
    drum_bars: list[dict[str, rhythm.Pattern]]   # [A, A'] - a two-bar riddim
    stab: rhythm.Pattern
    bass: bassline.BassParams
    bass_cells: list[rhythm.Pattern]              # [verse cell, peak cell]
    bass_pitch_seed: float
    lead_style: str
    lead: melody.LeadParams
    swing: float
    # One progression per section, in section order (may repeat).
    progressions: tuple = ()

    def describe(self) -> str:
        d = self.drums
        distinct = list(dict.fromkeys(p.name for p in self.progressions))
        return (
            f"drums={d.feel} kick@{sorted(d.kick_anchors)} dens={d.kick_density:.2f} "
            f"sync={d.kick_syncopation:.2f} | stabs {groove.describe(self.stab)} | "
            f"bass dens={self.bass.density:.2f} sync={self.bass.syncopation:.2f} "
            f"approach={self.bass.approach:.2f} contour={self.bass.contour} "
            f"answer={self.bass.answer_bars} | lead={self.lead_style} "
            f"{self.lead.contour} dens={self.lead.density:.2f} "
            f"span={int(self.lead.span)} oct={int(self.lead.octave):+d}"
            + (f" | harmony {' / '.join(distinct)}" if len(distinct) > 1 else "")
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def compose(
    profile: str | Profile | SongParams = "roots_dub",
    key: str | None = None,
    bars: int | None = None,
    seed: int | None = None,
    rig: Rig | None = None,
    template: str | None = None,
    progression: str | None = None,
    name: str = "",
    params: SongParams | None = None,
) -> Song:
    """Build a complete song. Deterministic for a given params/seed.

    Accepts either a `SongParams` (the real interface) or the older keyword
    form, which is turned into one.
    """
    if isinstance(profile, SongParams):
        params = profile
    if params is None:
        prof_name = profile.name if isinstance(profile, Profile) else profile
        params = SongParams(
            profile=profile_mod.get(prof_name).name,
            seed=random.randrange(1, 1_000_000) if seed is None else seed,
            key=key,
            progression=progression,
        )
        if bars is not None:
            params.structure = StructureParams(total_bars=bars)
        if template is not None:
            arr = structure.build(template, bars=bars)
            params.sections = [_section_spec(s) for s in arr.sections]

    prof = profile if isinstance(profile, Profile) else profile_mod.get(params.profile)
    rig = rig or Rig()
    rng = random.Random(params.seed)

    scale = _resolve_scale(params, prof, rng)
    arrangement = _resolve_structure(params, prof, rng)
    material = _resolve_material(params, prof, scale, arrangement, rng)

    # The resolved per-section progression names land on the sections too,
    # so the UI and the plan show what each section actually harmonises.
    for sec, prog in zip(arrangement.sections, material.progressions):
        sec.progression = prog.name

    song = Song(
        name=name or f"{prof.name}_{params.seed}",
        bpm=params.bpm,
        arrangement=arrangement,
        harmony=material.voiced,
        scale=scale,
        seed=params.seed,
    )
    song.params = params

    builders = {
        "drums": _build_drums, "perc": _build_perc, "sub": _build_sub,
        "chords": _build_chords, "pad": _build_pad, "lead": _build_lead,
    }
    for role in arrangement.roles_used():
        dest = rig.for_role(role)
        part = Part(
            role=role, name=role, output=dest.output, channel=dest.channel,
            preset=dest.preset, monophonic=dest.monophonic, comment=dest.comment,
        )
        part_rng = random.Random(params.seed_for(role))
        for section in arrangement.sections:
            if section.plays(role):
                builders[role](part, section, prof, material, part_rng)
        if part.monophonic:
            _enforce_monophony(part)
        _add_automation(part, role, arrangement, prof)
        if part.notes:
            song.parts.append(part)

    song.concept = material.describe()
    song.notes_for_player = _performance_notes(prof, material, arrangement, song)
    return song


def regenerate(song: Song, part: str, rig: Rig | None = None, rng=None) -> Song:
    """A new take on one part; every other part is byte-identical.

    `part` may also be "structure" or "harmony", which redraw those and
    clear the resolved values so they are generated afresh. A maker clicking
    re-roll wants fresh entropy (the default); a caller that must reproduce
    its output - the critic's revision loop - passes its own `rng`.
    """
    params = song.params.copy()
    params.reseed(part, rng)
    if part == "structure":
        # A redraw means a new shape, not the same shape with new lengths.
        params.sections = None
        params.structure = None
    elif part == "harmony":
        params.progression = None
        params.tension = None
        for spec in params.sections or []:
            spec.progression = None   # per-section draws get a fresh roll too
    return compose(params=params, rig=rig, name=song.name)


# --------------------------------------------------------------------------
# Resolution: fill in whatever the params left open, and write it back
# --------------------------------------------------------------------------

def _resolve_scale(params: SongParams, prof: Profile, rng: random.Random) -> Scale:
    if params.key:
        scale = parse_key(params.key)
        if len(params.key.split()) == 1:
            scale = Scale(scale.root, rng.choice(prof.modes))
    else:
        root = rng.choice([pitch.parse_note(n) for n in ("G1", "A1", "D1", "E1", "F1")])
        scale = Scale(root, rng.choice(prof.modes))
    # Anchor the tonic at the bottom of the sub register: the bassline is
    # built from degrees above the root and needs room to move up.
    root = pitch.transpose_into(scale.root, prof.sub.low, prof.sub.low + 11)
    scale = Scale(root, scale.name)
    params.key = f"{pitch.note_name(scale.root)} {scale.name}"
    params.bpm = params.bpm or prof.bpm
    if params.swing is None:
        params.swing = prof.swing
    return scale


def _section_spec(s: Section) -> SectionSpec:
    return SectionSpec(s.name, s.bars, s.energy, list(s.active),
                       s.dub_throw, s.direction,
                       progression=s.progression, ramp=s.ramp,
                       development=s.development)


def _resolve_structure(params: SongParams, prof: Profile, rng: random.Random) -> Arrangement:
    if params.sections:
        sections = []
        cursor = 0
        for i, spec in enumerate(params.sections):
            sections.append(Section(
                name=spec.name, bars=spec.bars, energy=spec.energy,
                active=tuple(r for r in structure.ROLES if r in spec.active),
                # -1 on the spec means "follow the schedule": the ladder
                # climbs to the song's middle and comes home, so the outro
                # restates the statement instead of premiering the retrograde.
                development=(i if spec.development is None or spec.development < 0
                             else spec.development),
                direction=spec.direction, start_bar=cursor,
                dub_throw=spec.dub_throw,
                progression=spec.progression, ramp=spec.ramp,
            ))
            cursor += spec.bars
        return Arrangement(sections, "edited")

    # Structure has its own seed so it can be redrawn on its own, exactly
    # like a part.
    srng = random.Random(params.seed_for("structure"))
    if params.structure is None:
        params.structure = structure.params_for_family(prof.family, srng)
    arrangement = structure.generate(params.structure, srng, name=prof.family)
    params.sections = [_section_spec(s) for s in arrangement.sections]
    return arrangement


def _resolve_material(
    params: SongParams, prof: Profile, scale: Scale, arrangement: Arrangement,
    rng: random.Random,
) -> Material:
    # Harmony, from its own seed.
    hrng = random.Random(params.seed_for("harmony"))
    if params.tension is None:
        params.tension = round(hrng.uniform(
            max(0.05, prof.progression_tension - 0.2),
            min(0.95, prof.progression_tension + 0.25)), 2)
    if params.bars_per_chord is None:
        params.bars_per_chord = hrng.choice(prof.bars_per_chord_choices)

    sections = arrangement.sections
    if params.progression and params.progression != "generate":
        # A named progression applies to the whole song - that is what was
        # asked for; a user's choice is not a suggestion to vary.
        if params.progression in harmony.BY_NAME:
            prog = harmony.BY_NAME[params.progression]
        else:
            prog = _parse_roman(params.progression, scale)
        progs = [prog] * len(sections)
        core = prog
    else:
        progs, core = _harmony_for_sections(scale, sections, params, hrng)

    # One chord plan per section, laid out at the section's absolute start,
    # with voice leading continuing across the seams so a progression change
    # moves by the smallest step instead of every voice jumping to home.
    plan_pieces: list[harmony.HarmonyPlan] = []
    voiced_pieces: list[harmony.HarmonyPlan] = []
    previous: chord_mod.Chord | None = None
    for sec, prog in zip(sections, progs):
        piece = harmony.realise(
            scale, prog, sec.bars, params.bars_per_chord, size=prof.chord_size,
        ).offset(sec.start_tick)
        led = harmony.voiced(
            piece, prof.chords.low, prof.chords.high,
            rootless=True, centre=prof.chords.centre, first=previous,
        )
        if led.events:
            previous = led.events[-1].chord
        plan_pieces.append(piece)
        voiced_pieces.append(led)
    plan = harmony.merge_section_plans(plan_pieces, core)
    voiced = harmony.merge_section_plans(voiced_pieces, core)

    # Drums: a feel varied into a specific spec.
    if params.drums is None:
        base = groove.FEELS.get(prof.drum_style, groove.HALF_TIME)
        params.drums = groove.vary(base, rng, amount=0.15)
    drum_rng = random.Random(params.seed_for("drums"))
    bar_a = groove.drums(params.drums, drum_rng)
    bar_b = groove.drums(groove.vary(params.drums, drum_rng, 0.1), drum_rng)

    # Stabs.
    feel = prof.feel
    if params.stab_density is None:
        params.stab_density = round(rng.uniform(0.5, 1.0), 2)
    if params.stab_syncopation is None:
        params.stab_syncopation = round(rng.uniform(0.6, 0.95), 2)
    stab = groove.generate(
        groove.stab_spec(feel, params.stab_density, params.stab_syncopation,
                         note_length=prof.skank_length),
        random.Random(params.seed_for("chords")),
    )

    # Bass.
    if params.bass is None:
        params.bass = bassline.params_for_feel(feel, rng, energy=0.6)
    bass_rng = random.Random(params.seed_for("sub"))
    cells = [
        groove.generate(params.bass.groove(), bass_rng),
        # The peak cell is a little busier and more syncopated - but only a
        # little. A sub that doubles its note count at the drop stops being
        # a sub.
        groove.generate(replace(params.bass, density=min(0.6, params.bass.density + 0.04),
                                syncopation=min(1.0, params.bass.syncopation + 0.1)).groove(),
                        bass_rng),
    ]

    # Lead. The params are drawn from the *shared* rng, exactly the bass's
    # pattern (params.bass = bassline.params_for_feel(feel, rng, ...)): one
    # resolution draw, written back, so a recompose reads them instead of
    # redrawing. The bass's own stream must stay put - whatever the lead
    # resolves, it may not shift bass_rng, or the pitch seed drawn from it
    # would differ between a first compose and a recompose, and the song
    # would stop reproducing. lead_rng - the lead's own seed - is left to the
    # motif alone, so a re-roll of the lead keeps the character (the params)
    # and draws a new line, the same way a re-roll of the sub works.
    if params.lead_style is None:
        params.lead_style = rng.choice(prof.lead_styles)
    if params.lead is None:
        params.lead = melody.LeadParams(
            density=rng.uniform(0.18, 0.45),
            syncopation=rng.uniform(0.0, 0.7),
            span=rng.choice((4, 5, 6, 7, 8, 9)),
            octave=rng.choice((-1, 0, 0, 1)),
            step_bias=rng.uniform(0.5, 0.85),
            ornament=rng.uniform(0.0, 0.5),
            contour=rng.choice(sorted(melody.CONTOURS)),
        )
    lead = params.lead
    lead_rng = random.Random(params.seed_for("lead"))
    # The param values, resolved into the motif: density into pulse count,
    # syncopation into rotation, octave into the centre of the range.
    pulses = max(2, int(round(lead.density * STEPS_PER_BAR)))
    rotation = int(round(lead.syncopation * 8))
    motif_rhythm = rhythm.euclid(pulses, STEPS_PER_BAR, rotation=rotation)
    motif = melody.generate(
        motif_rhythm,
        contour=lead.contour,
        span=int(lead.span), centre=int(lead.octave) * scale.size,
        chord=plan.events[0].chord if plan.events else None,
        scale=scale, rng=lead_rng, name="motif",
        step_bias=lead.step_bias, ornament_amount=lead.ornament,
    )

    return Material(
        scale=scale, plan=plan, voiced=voiced, motif=motif,
        drums=params.drums, drum_bars=[bar_a, bar_b], stab=stab,
        bass=params.bass, bass_cells=cells, bass_pitch_seed=bass_rng.random(),
        lead_style=params.lead_style, lead=params.lead, swing=params.swing or 0.0,
        progressions=tuple(progs),
    )


# The energy boundaries that give a section its own harmonic personality:
# at or above this a section is a peak and earns a cadence; at or below it a
# hole and earns a rest chord; between, it sits on the tune's core.
PEAK_ENERGY = 0.8
HOLE_ENERGY = 0.35


def _progression_by_name(name: str, scale: Scale) -> harmony.Progression:
    if name in harmony.BY_NAME:
        return harmony.BY_NAME[name]
    return _parse_roman(name, scale)


def _harmony_for_sections(
    scale: Scale,
    sections: list[Section],
    params: SongParams,
    hrng: random.Random,
) -> tuple[list[harmony.Progression], harmony.Progression]:
    """One progression per section, three personalities in all.

    Body sections share the tune's core, generated at the global tension.
    Peaks (energy >= PEAK_ENERGY) share one cadence drawn from the library at
    high tension - this is where a drop earns its own identity, and sharing
    it between two drops is what makes the second one read as a return.
    Holes (energy <= HOLE_ENERGY - intros, dub-outs, breakdowns, outros)
    share one rest chord, because a hole's harmony is a place to go, not a
    destination.

    Section specs that already carry a progression name (resolved on a
    previous compose, or set by the user or the UI) are honoured exactly;
    only the open sections draw. Draws are written back into the specs so a
    recompose reproduces the song to the bit.

    Returns the per-section progression list and the core - the progression
    of the first mid-energy section, since that is what the tune *is*; the
    song header reports the core, not the drone that usually opens it.
    """
    base = peak = release = None
    progs: list[harmony.Progression] = []
    for i, sec in enumerate(sections):
        spec = params.sections[i] if params.sections and i < len(params.sections) else None
        if spec is not None and spec.progression:
            progs.append(_progression_by_name(spec.progression, scale))
            continue
        if sec.energy >= PEAK_ENERGY:
            if peak is None:
                peak = harmony.choose(scale.name, 0.85, hrng, tolerance=0.4)
            progs.append(peak)
        elif sec.energy <= HOLE_ENERGY:
            if release is None:
                release = harmony.choose(scale.name, 0.15, hrng)
            progs.append(release)
        else:
            if base is None:
                base = harmony.generate_progression(scale, params.tension, hrng)
            progs.append(base)
        if spec is not None:
            spec.progression = progs[-1].name
    core = next(
        (p for s, p in zip(sections, progs)
         if HOLE_ENERGY < s.energy < PEAK_ENERGY),
        progs[0],
    )
    return progs, core


def _parse_roman(text: str, scale: Scale) -> harmony.Progression:
    """Turn 'i-bVII-bVI-V' back into a Progression for this scale."""
    names = [n.strip() for n in text.replace(" ", "-").split("-") if n.strip()]
    degrees, majors = [], []
    for token in names:
        matched = None
        for d in range(scale.size):
            base = harmony.roman(scale, d)
            raised = harmony.roman(scale, d, chord_mod.with_major_third(chord_mod.triad(scale, d)))
            if token == base:
                matched = d
            elif token == raised:
                matched = d
                majors.append(d)
            if matched is not None:
                break
        if matched is None:
            # The spelling of a degree depends on the mode - 'bVII' is valid
            # in natural minor but that degree is vii° in harmonic minor - so
            # tell the writer what this key actually calls each degree.
            spellings = ", ".join(harmony.roman(scale, d)
                                   for d in range(scale.size))
            raise ValueError(
                f"cannot read {token!r} in {scale} (this key spells: {spellings})")
        degrees.append(matched)
    return harmony.Progression(text, tuple(degrees), (scale.name,), 0.5,
                               "specified", tuple(majors))


# --------------------------------------------------------------------------
# Part builders
# --------------------------------------------------------------------------

def _emit(part: Part, notes, offset_ticks: int, low=None, high=None) -> None:
    for midi, step, length, vel in notes:
        if low is not None and high is not None:
            midi = pitch.transpose_into(midi, low, high)
        if not 0 <= midi <= 127:
            continue
        start = offset_ticks + int(round(step * TICKS_PER_16TH))
        ticks = max(1, int(round(length * TICKS_PER_16TH)))
        part.add(Note(midi, start, ticks, max(1, min(127, int(vel)))))


def _build_drums(part, section, prof, m: Material, rng):
    for bar in range(section.bars):
        bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
        # A ramped section climbs bar by bar; a flat one sits on its level.
        energy = structure.bar_energy(section, bar)
        density = structure.density_for(energy, "drums")
        # A two-bar riddim: A A A A' - the variant on every fourth bar.
        voices = m.drum_bars[1] if (bar % 4 == 3) else m.drum_bars[0]
        ends_phrase = (bar + 1) % 8 == 0
        ends_section = bar == section.bars - 1
        # The last bar of a section is a fill: the snare and rim let go of
        # the riddim and run into the next section's downbeat. An 8-bar
        # phrase end inside a section just thickens the ordinary pattern.
        fill_this_bar = ends_section and section.bars >= 4

        for voice, pattern in voices.items():
            if voice not in KIT_ROWS or not pattern.hits:
                continue
            if fill_this_bar and voice in ("snare", "rim"):
                continue  # the fill owns this bar
            keep = density if voice in ("hat", "rim") else min(1.0, density + 0.35)
            p = rhythm.thin(pattern, keep, rng)
            if m.swing:
                p = rhythm.swing(p, m.swing)
            if voice == "snare":
                p = rhythm.lay_back(p, 0.05, tags=())
            p = rhythm.humanize(p, 0.02, 6, rng)
            if ends_phrase and not fill_this_bar and voice == "snare":
                p = rhythm.thicken(p, 3, rng, velocity=95)
            vel_scale = structure.velocity_for(energy) / 100.0
            _emit(part, [(KIT_ROWS[voice], h.step, min(h.length, 0.9), h.velocity * vel_scale)
                         for h in p.sorted_hits()], bar_tick)

        if fill_this_bar:
            vel_scale = structure.velocity_for(energy) / 100.0
            for voice, pattern in fill.make(energy, rng).items():
                if voice not in KIT_ROWS:
                    continue
                _emit(part, [(KIT_ROWS[voice], h.step, min(h.length, 0.9),
                              h.velocity * vel_scale) for h in pattern.sorted_hits()],
                      bar_tick)


def _build_perc(part, section, prof, m: Material, rng):
    for bar in range(0, section.bars, 2):
        energy = structure.bar_energy(section, bar)
        density = structure.density_for(energy, "perc")
        pulses = max(2, int(round(3 + density * 5)))
        # A fresh cell per two bars: a shaker that loops one figure for a
        # whole section is the sound of a loop, not of a player.
        pattern = rhythm.euclid(pulses, STEPS_PER_BAR, rotation=rng.choice((1, 3, 5)))
        pattern = rhythm.accent(pattern, strong=88, weak=58)
        if m.swing:
            pattern = rhythm.swing(pattern, m.swing)
        two_bar = rhythm.every_other_bar(pattern, rhythm.rotate(pattern, 2), bars=2)
        bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
        shaken = rhythm.humanize(two_bar, 0.03, 8, rng)
        vel_scale = structure.velocity_for(energy) / 100.0
        _emit(part, [(KIT_ROWS["shaker"], h.step, 0.5, h.velocity * vel_scale)
                     for h in shaken.sorted_hits()], bar_tick)


def _build_sub(part, section, prof, m: Material, rng):
    """The bassline: generated phrases, cell restated and answered.

    The cell rhythm is fixed for the song (it is the bassline's identity)
    and its pitch choices are replayed from a fixed seed, so every
    restatement is a restatement. Answer bars draw fresh randomness. Peaks
    use the second, busier cell; a 32-bar peak switches cell halfway.
    """
    params = m.bass
    density = structure.density_for(section.energy, "sub")
    phrase_len = params.phrase_bars
    is_peak = section.energy >= 0.85
    fill_params = replace(
        params, density=min(0.6, params.density + params.fill_density_boost),
        syncopation=min(1.0, params.syncopation + 0.15),
        approach=min(1.0, params.approach + 0.2),
    )

    for start in range(0, section.bars, phrase_len):
        bars = min(phrase_len, section.bars - start)
        roots = []
        for b in range(bars):
            ev = m.plan.at((section.start_bar + start + b) * TICKS_PER_BAR)
            roots.append(ev.chord.degree if ev else 0)

        cell = m.bass_cells[1] if is_peak else m.bass_cells[0]
        if is_peak and section.bars >= 32 and start >= section.bars // 2:
            cell = m.bass_cells[0]
        fill_rhythm = groove.generate(fill_params.groove(), rng)

        motifs = []
        for b in range(bars):
            root, nxt = roots[b], roots[(b + 1) % bars]
            if b in params.answer_bars and bars == phrase_len:
                motifs.append(bassline.generate_line(fill_rhythm, root, rng, fill_params, nxt))
            else:
                motifs.append(bassline.generate_line(
                    cell, root, random.Random(m.bass_pitch_seed), params, nxt))

        # Phrase-level colour. A two-note root-and-fifth cell is authentic,
        # but four bars of nothing else is inert - so if the phrase has fewer
        # than three pitch classes, the answer bar is redrawn to lean on
        # approach and colour tones until it supplies them.
        def pcs(ms):
            return {m.scale.note(tn.degree) % 12 for mo in ms for tn in mo.tones}
        answers = [b for b in params.answer_bars if b < bars]
        if answers and len(pcs(motifs)) < 3:
            colour_params = replace(
                fill_params, approach=0.0, fifth_weight=0.9, octave_jumps=0.3,
                density=min(0.6, fill_params.density + 0.06))
            for _ in range(6):
                b = answers[-1]
                motifs[b] = bassline.generate_line(
                    groove.generate(colour_params.groove(), rng), roots[b], rng,
                    colour_params, roots[(b + 1) % bars])
                if len(pcs(motifs)) >= 3:
                    break

        for b, motif in enumerate(motifs):
            abs_bar = section.start_bar + start + b
            if density < 0.85:
                motif = melody.thin_tones(motif, density, rng)
            notes = bassline.constrain_register(motif, m.scale, prof.sub.low, prof.sub.high)
            notes = bassline.monophonic(notes)
            notes = [(mi, s, l * prof.bass_note_length,
                      structure.velocity_for(section.energy, v)) for mi, s, l, v in notes]
            _emit(part, notes, abs_bar * TICKS_PER_BAR)


def _build_chords(part, section, prof, m: Material, rng):
    for bar in range(section.bars):
        bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
        ev = m.voiced.at(bar_tick)
        if ev is None:
            continue
        density = structure.density_for(structure.bar_energy(section, bar), "chords")
        p = rhythm.thin(m.stab, density, rng) if density < 1.0 else m.stab
        if m.swing:
            p = rhythm.swing(p, m.swing)
        p = rhythm.humanize(p, 0.02, 7, rng)
        vel = lambda v: structure.velocity_for(
            structure.bar_energy(section, bar), v)  # noqa: E731
        for hit in p.sorted_hits():
            for note in ev.chord.pitches:
                _emit(part, [(note, hit.step, hit.length, vel(hit.velocity))], bar_tick)


def _build_pad(part, section, prof, m: Material, rng):
    start, end = section.start_tick, section.start_tick + section.length_ticks
    for ev in m.voiced.events:
        if ev.end <= start or ev.start >= end:
            continue
        s, e = max(ev.start, start), min(ev.end, end)
        if e - s <= 0:
            continue
        vel = structure.velocity_for(section.energy, 66)
        for note in ev.chord.pitches:
            placed = pitch.transpose_into(note, prof.pad.low, prof.pad.high)
            part.add(Note(placed, s, max(1, e - s - TICKS_PER_16TH), vel))


def _build_lead(part, section, prof, m: Material, rng):
    """The motif, developed to this section's stage, in one of three styles.

    All three are sparse on purpose - a dub lead is a comment on the riddim,
    not a solo - but each guarantees the lead is heard at least every other
    bar while it is active.
    """
    style = m.lead_style
    developed, _ = melody.develop(m.motif, section.development, rng, scale=m.scale)
    density = structure.density_for(section.energy, "lead")
    velocity = lambda v: structure.velocity_for(section.energy, v)  # noqa: E731

    if style == "long_tones":
        degrees = developed.degrees() or [0]
        per_bar = 2 if density >= 0.7 else 1
        hold = (STEPS_PER_BAR // per_bar) - 2
        for bar in range(section.bars):
            bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
            ev = m.plan.at(bar_tick)
            for slot in range(per_bar):
                degree = degrees[(bar * per_bar + slot) % len(degrees)]
                if ev is not None:
                    degree += ev.chord.degree
                    # The first tone of the bar agrees with the chord; the
                    # second is free to be a scale tone, or four bars of
                    # long tones collapse onto one note.
                    if slot == 0:
                        degree = melody.nearest_chord_degree(degree, ev.chord, m.scale)
                # Long tones do not read the motif's velocity shape; give
                # them their own, with a little humanize so they do not stack
                # at one level.
                _emit(part, [(m.scale.note(degree), slot * (STEPS_PER_BAR // per_bar),
                              hold, velocity(78 + rng.randint(-8, 8)))],
                      bar_tick, low=prof.lead.low, high=prof.lead.high)
        return

    if style == "riff":
        phrase, gap = melody.sequence(developed, [0, -1]), (4 if density >= 0.6 else 6)
    else:
        phrase, gap = developed, (2 if density >= 0.55 else 3)

    for index, bar in enumerate(range(0, section.bars, gap)):
        bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
        ev = m.plan.at(bar_tick)
        statement = phrase
        if style == "call_response" and index % 2 == 1:
            statement = melody.transpose(statement, -1)
        if ev is not None:
            statement = melody.transpose(statement, ev.chord.degree)
        room = (section.bars - bar) * STEPS_PER_BAR
        notes = [(mi, s, min(l, room - s), velocity(v))
                 for mi, s, l, v in statement.notes(m.scale) if s < room]
        _emit(part, notes, bar_tick, low=prof.lead.low, high=prof.lead.high)


def _enforce_monophony(part: Part) -> None:
    """No two notes in this part ever sound at once - one oscillator."""
    notes = part.sorted_notes()
    fixed: list[Note] = []
    for note, following in zip(notes, notes[1:] + [None]):
        length = note.length
        if following is not None:
            length = min(length, following.start - note.start)
        if length >= 1:
            fixed.append(Note(note.midi, note.start, length, note.velocity))
    part.notes = fixed


def _add_automation(part: Part, role: str, arrangement: Arrangement, prof: Profile):
    """Filter and reverb moves following the energy curve - synth clips only,
    because the firmware rejects param automation on anything else."""
    if part.output != "synth":
        return
    for i, section in enumerate(arrangement.sections):
        if not section.plays(role):
            continue
        brightness = structure.brightness_for(section.energy)
        if section.ramp > 0:
            # A build opens inside the section, the same bar-by-bar way the
            # density ramps - the lid comes off before the drop, not at it.
            part.automation.append(Automation(
                "lpf", section.start_tick, section.length_ticks,
                structure.brightness_for(structure.bar_energy(section, 0)),
                brightness))
        else:
            following = (arrangement.sections[i + 1]
                         if i + 1 < len(arrangement.sections) else None)
            if following is not None and following.energy - section.energy > 0.3:
                part.automation.append(Automation(
                    "lpf", section.start_tick, section.length_ticks, brightness,
                    min(1.0, structure.brightness_for(following.energy))))
            else:
                part.automation.append(Automation(
                    "lpf", section.start_tick, section.length_ticks, brightness))
        if section.dub_throw:
            part.automation.append(Automation(
                "reverb", section.start_tick, section.length_ticks, 0.55, 0.8))


def _performance_notes(prof: Profile, m: Material, arrangement: Arrangement, song: Song):
    throws = [s.name for s in arrangement.sections if s.dub_throw]
    midi_parts = sorted((p for p in song.parts if p.output == "midi"), key=lambda p: p.channel)
    automated = any(p.automation for p in song.parts)
    channels = ", ".join(f"{p.role} ch{p.channel}" for p in midi_parts)
    return [
        f"Key: {m.scale}. Progression: {m.plan.progression.name} "
        f"({' '.join(m.voiced.symbols()[:4])}...).",
        f"Bass: density {m.bass.density:.2f}, syncopation {m.bass.syncopation:.2f}, "
        f"approach tones {m.bass.approach:.2f}, contour {m.bass.contour}"
        f"{', anticipates the barline' if m.bass.anticipate else ''}.",
        f"Coral sub voice: single osc, {prof.sub}. Keep it mono and the filter "
        f"low; everything above {pitch.note_name(prof.chords.low)} is left for the chords.",
        (f"MIDI out: {channels}. Drums and perc stay on Deluge kits." if channels
         else "Everything is on Deluge instruments."),
        f"Nostalgia: chord stabs on the delay send. Throw going into: {', '.join(throws)}.",
        "Pam's: the arrangement assumes Deluge is clock master.",
        ("Filter automation is written into Deluge synth clips only; it is a floor, "
         "not a performance." if automated else
         "No filter automation is written - every pitched part is on the rack. "
         "All filter movement is yours; the energy curve is in the notes and mutes."),
    ]

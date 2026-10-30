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

import math
import random
from dataclasses import dataclass, field, replace

from . import profile as profile_mod
from .arrange import dynamics, interplay, structure
from .arrange.dynamics import DynamicsParams
from .arrange.interplay import InterplayParams, Line
from .arrange.structure import Arrangement, Section, StructureParams
from .params import PARTS, SectionSpec, SongParams
from .profile import Profile
from .song import Automation, Note, Part, Rig, Song
from .theory import (
    bassline, chord as chord_mod, fill, groove, harmony, hook as hook_mod,
    melody, percussion, pitch, rhythm,
)
from .theory import cells as mined_cells
from .theory.rhythm import metric_weight
from .theory.scale import Scale, parse_key
from .units import STEPS_PER_BAR, TICKS_PER_16TH, TICKS_PER_BAR

# Deluge kit rows: note 36 is the bottom visible row, one semitone per row up.
# Crash and riser are the tension rows: load a crash (or impact) on 42 and a
# noise sweep or reverse cymbal on 43 of the drum kit.
KIT_ROWS = {"kick": 36, "snare": 37, "hat": 38, "rim": 39, "perc": 40, "shaker": 41,
            "crash": 42, "riser": 43,
            # The hands, on the perc kit: see theory/percussion.py.
            "conga_lo": 44, "conga_hi": 45, "bongo": 46, "block": 47, "tamb": 48,
            "tom_lo": 49, "tom_hi": 50,
            # The sound system's siren, on the drum kit (Profile.sirens).
            "siren": 51}
HAND_ROWS = tuple(KIT_ROWS[v] for v in ("conga_lo", "conga_hi", "bongo", "block", "tamb",
                                        "tom_lo", "tom_hi"))


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
    # The theme and the conversation around it.
    hook: hook_mod.Hook | None = None
    lines: dict = field(default_factory=dict)       # section start_bar -> Line
    drum_rockers: list = field(default_factory=list)   # [R, R'] - see ROCKERS_BLOCKS
    interplay: InterplayParams = InterplayParams()
    dynamics: DynamicsParams = DynamicsParams()
    lead_shift: int = 0                             # semitones: hook -> lead window
    lead_seed: int = 0                              # the lead's own performance stream
    drum_peak: list = field(default_factory=list)   # [P, P'] - the riddim at full weight
    echo: tuple = (0.0, 3.0)                        # the hook's rhythm, as an answer
    sections: list = field(default_factory=list)
    silences: list = field(default_factory=list)    # (start, end, roles that keep playing)
    mutes: list = field(default_factory=list)       # (start, end, {(role, row|None)}) held back
    built: dict = field(default_factory=dict)       # parts written so far, by role
    lift: tuple = (0, 0)                            # (from tick, semitones): the last drop's key lift
    seeds: dict = field(default_factory=dict)       # each part's own seed, by role
    hands: object = None                            # percussion.PercParams, drawn with the perc

    def neighbours(self, section: Section) -> tuple[Section | None, Section | None]:
        i = next((k for k, s in enumerate(self.sections) if s.start_bar == section.start_bar), None)
        if i is None:
            return None, None
        prev = self.sections[i - 1] if i > 0 else None
        nxt = self.sections[i + 1] if i + 1 < len(self.sections) else None
        return prev, nxt

    def line(self, section: Section) -> Line:
        return self.lines.get(section.start_bar) or Line()

    def describe(self) -> str:
        d = self.drums
        distinct = list(dict.fromkeys(p.name for p in self.progressions))
        return (
            f"drums={d.feel} kick@{sorted(d.kick_anchors)} dens={d.kick_density:.2f} "
            f"sync={d.kick_syncopation:.2f} | stabs {groove.describe(self.stab)} | "
            f"bass dens={self.bass.density:.2f} sync={self.bass.syncopation:.2f} "
            f"approach={self.bass.approach:.2f} contour={self.bass.contour} "
            f"answer={self.bass.answer_bars} | lead={self.lead_style} "
            f"gate={self.lead.gate:.2f} slide={self.lead.slide:.2f} "
            f"accent={self.lead.accent:.2f} oct={int(self.lead.octave):+d}"
            + (f" | hook {self.hook.params.form} [{self.hook.form_string()}] "
               f"{self.hook.params.rhythm} {self.hook.contour} "
               f"score={self.hook.score:.2f}" if self.hook else "")
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
    bpm: int | None = None,
    live: bool = False,
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
            bpm=bpm,
        )
        if bars is not None:
            params.structure = StructureParams(total_bars=bars)
        if template is not None:
            arr = structure.build(template, bars=bars)
            params.sections = [_section_spec(s) for s in arr.sections]

    if live:
        params = _make_live(params)
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
    # Built in listening order: the lead first (everyone makes room for what
    # it actually plays), then the bass (which the kick locks to), then the
    # rest. Parts are listed in the usual order.
    order = ("lead", "sub", "chords", "pad", "drums", "perc")
    used = arrangement.roles_used()
    built: dict[str, Part] = {}
    material.built = built
    for role in sorted(used, key=lambda r: order.index(r) if r in order else 99):
        dest = rig.for_role(role)
        part = Part(
            role=role, name=role, output=dest.output, channel=dest.channel,
            preset=dest.preset, monophonic=dest.monophonic, comment=dest.comment,
        )
        if part.output == "kit" and not part.preset:
            part.preset = profile_mod.KIT_PRESETS.get(prof.family, "")
        part_rng = random.Random(params.seed_for(role))
        material.seeds[role] = params.seed_for(role)
        for section in arrangement.sections:
            if section.plays(role):
                builders[role](part, section, prof, material, part_rng)
        _apply_silences(part, material.silences)
        _apply_mutes(part, material.mutes)
        if part.output == "kit" and role == "drums":
            _thin_kick_in_pulls(part, material.mutes)
        # The lead is a sung line whatever synth it lands on: one note at a
        # time, so an ornament can never stack onto the note it decorates.
        if part.monophonic:
            _enforce_monophony(part)
        elif role == "lead":
            _lead_line(part)
        _add_automation(part, role, arrangement, prof)
        _add_kit_automation(part, role, arrangement, prof)
        built[role] = part
    for role in used:
        if built[role].notes:
            song.parts.append(built[role])

    _clamp_to_song(song, arrangement)
    _dub_throws(song, arrangement, material, params)
    _phrase_dynamics(song, arrangement)
    _lift(song, material, prof)
    _button(song, arrangement, material, prof)
    song.concept = material.describe()
    song.notes_for_player = _performance_notes(prof, material, arrangement, song)
    return song


def _clamp_to_song(song: Song, arrangement) -> None:
    """Nothing rings past the arrangement: a pickup in the last bar leans
    into the final hit (the coda) rather than over it."""
    end = arrangement.length_ticks
    for part in song.parts:
        part.notes = [n if n.end <= end else replace(n, length=end - n.start)
                      for n in part.notes if n.start < end]


def _dub_throws(song: Song, arrangement, m: Material, params: SongParams) -> None:
    """The engineer's throws in a dub-out: with the band stripped to drums
    and bass, a lone skank is let through every couple of bars - one stab on
    an offbeat, short and hard, so the delay send (the chords' channel sits
    on it) is what fills the bar - and at the end of each eight, a scrap of
    the hook's head on the lead. The dub sound is the echo of what is not
    being played."""
    chords = next((p for p in song.parts if p.role == "chords"), None)
    lead = next((p for p in song.parts if p.role == "lead"), None)
    for i, sec in enumerate(arrangement.sections):
        if not sec.name.startswith("dub_out") or sec.plays("chords"):
            continue
        rng = random.Random(f"{params.seed}:throws:{i}")
        thrown = set()
        for bar in range(1, sec.bars, 2):
            if bar == sec.bars - 1:
                continue                       # the bar before the next entry stays clear
            tick = (sec.start_bar + bar) * TICKS_PER_BAR
            if chords is not None and rng.random() < 0.75:
                ev = m.voiced.at(tick)
                if ev is not None:
                    st = rng.choice((6, 14)) if bar % 4 == 1 else rng.choice((2, 10))
                    for n in ev.chord.pitches:
                        chords.add(Note(n, tick + st * TICKS_PER_16TH, TICKS_PER_16TH // 2 + 6, 104))
                    thrown.add("chords")
            if lead is not None and m.hook is not None and bar % 8 == 7 and sec.bars >= 8:
                head = [t for t in m.hook.cell.tones if t.step < 6][:3]
                base = head[0].step if head else 0
                for t in head:
                    lead.add(Note(t.midi(m.scale) + m.lead_shift,
                                  tick + int((4 + t.step - base) * TICKS_PER_16TH),
                                  TICKS_PER_16TH, 92))
                if head:
                    thrown.add("lead")
        # The section now features what it throws - so the Deluge gets a clip
        # for it - while the song's recipe (its section specs) is untouched:
        # a recompose builds the dub-out bare and throws again.
        if thrown:
            sec.active = tuple(r for r in structure.ROLES if r in set(sec.active) | thrown)


# Loudness across an eight-bar phrase: lean on its first downbeat, ease off,
# hold, then push into the turnaround. Small - a player's shaping, not an
# automation ride.
_PHRASE_ARC = (1.05, 0.96, 0.99, 0.98, 1.02, 0.97, 1.01, 1.07)


def _phrase_dynamics(song: Song, arrangement) -> None:
    """Shape the time-keepers' velocity by where they sit in the phrase.

    Hats, shaker, ghost snares and the chord stabs breathe with the phrase:
    the first bar leans in, the middle relaxes, the eighth bar pushes, and
    across the phrase the hats open up a little. The kick, the backbeat and
    the sub do not move - they are the floor. Timing is never touched."""
    hat, shaker, snare = KIT_ROWS["hat"], KIT_ROWS["shaker"], KIT_ROWS["snare"]
    for part in song.parts:
        if part.role not in ("drums", "perc", "chords"):
            continue
        out = []
        for n in part.notes:
            sec = arrangement.at_tick(n.start)
            keeper = part.role == "chords" or n.midi in (hat, shaker) or \
                (n.midi == snare and n.velocity < 90)      # ghosts, not the backbeat
            if sec is None or not keeper or sec.bars < 8:
                out.append(n)
                continue
            rel = (n.start - sec.start_tick) // TICKS_PER_BAR
            pos = rel % 8
            depth = 0.5 + 0.5 * sec.energy           # peaks breathe more
            f = 1.0 + (_PHRASE_ARC[pos] - 1.0) * depth
            if n.midi in (hat, shaker) and part.role != "chords":
                f *= 1.0 + 0.012 * pos * depth       # the hats open across the phrase
                within = (n.start % TICKS_PER_BAR) / TICKS_PER_BAR
                if pos == 7:
                    f *= 1.0 + 0.08 * within         # and swell through the last bar
            out.append(replace(n, velocity=max(1, min(127, int(round(n.velocity * f))))))
        part.notes = out


def _lift(song: Song, m: Material, prof) -> None:
    """Move everything pitched from the lift point to the end up by the
    lift, and the harmony with it, each part kept inside its register (a
    sub that would leave its window drops the octave; so does a lead or a
    chord tone)."""
    start, k = m.lift
    if not k:
        return
    windows = {"sub": (prof.sub.low, prof.sub.high), "chords": (prof.chords.low, prof.chords.high),
               "pad": (prof.pad.low, prof.pad.high),
               "lead": (lead_window(prof, m.lead)[0], lead_window(prof, m.lead)[1] + 5)}
    for part in song.parts:
        if part.output == "kit" or part.role not in windows:
            continue
        lo, hi = windows[part.role]
        after = [n for n in part.notes if n.start >= start]
        shift = k
        if part.role in ("lead", "sub") and after and max(n.midi for n in after) + k > hi \
                and min(n.midi for n in after) + k - 12 >= lo:
            shift = k - 12          # a line moves as one: the whole passage drops the octave
        moved = []
        for n in part.notes:
            if n.start >= start:
                p = n.midi + shift
                if p > hi:          # chord tones fold to the next inversion down
                    p -= 12
                n = replace(n, midi=max(lo, p))
            moved.append(n)
        part.notes = moved
    new_scale = Scale(m.scale.root + k, m.scale.name)
    events = []
    for ev in song.harmony.events:
        if ev.start >= start:
            c = chord_mod.Chord(tuple(p + k for p in ev.chord.pitches), ev.chord.degree, new_scale)
            ev = chord_mod.ChordEvent(c, ev.start, ev.length, c.symbol(), dict(ev.meta, lift=k))
        events.append(ev)
    song.harmony = harmony.HarmonyPlan(song.harmony.scale, song.harmony.progression, events)


def _make_live(params: SongParams) -> SongParams:
    """The same song as a live track: its structure redrawn as short loops
    straight in at energy (see structure._live). Everything else - hook,
    harmony, groove, sound - is kept."""
    if params.structure is not None and params.structure.live:
        return params
    params = params.copy()
    prof = profile_mod.get(params.profile)
    base = params.structure or structure.params_for_family(
        prof.family, random.Random(params.seed_for("structure")), params.bpm or prof.bpm)
    # About three minutes, the measured sets' tune length at any tempo (~100
    # bars at 140, ~55 at 72), and the desk pulling the bass at the sets'
    # rate - 0.7 to 0.9 a 32 bars in every family, where the dub records
    # alone pull more (references/stats/sets.md).
    lrng = random.Random(f"{params.seed}:live")
    bpm = params.bpm or prof.bpm
    params.structure = replace(base, live=True,
                               live_bars=max(48, round(lrng.uniform(2.5, 3.5) * bpm / 4)))
    if params.dynamics is None:
        params.dynamics = dynamics.params_for(
            prof.family, random.Random(f"{params.seed}:dynamics"), LIVE_PULLS)
    else:
        params.dynamics = replace(params.dynamics,
                                  bass_drops=round(lrng.uniform(*LIVE_PULLS), 2))
    params.sections = None
    return params


LIVE_PULLS = (0.7, 0.9)


def bass_style_pool(prof: Profile) -> tuple:
    """The kinds of bassline a profile draws from, with their weights."""
    pool = {"dubstep": (("sustain", 3), ("walk", 3), ("bounce", 2), ("sparse", 2),
                        ("skip", 2)),
            "garage": (("skip", 4), ("walk", 3), ("bounce", 2), ("sustain", 1)),
            "dub": (("walk", 4), ("sparse", 2), ("bounce", 2), ("sustain", 1), ("pulse", 1))}
    if prof.name == "steppers_dub":
        return (("pulse", 3), ("walk", 3), ("bounce", 2))
    if prof.name == "dark_garage":
        return (("skip", 4), ("walk", 3), ("sustain", 2), ("bounce", 1))
    return pool.get(prof.family, pool["dub"])


# Bass styles whose notes come from the sub's own seed; the others are fixed
# shapes over the chords, so a fresh seed alone would not change a note.
SEEDED_BASS = ("walk", "sparse")


def regenerate(song: Song, part: str, rig: Rig | None = None, rng=None) -> Song:
    """A new take on one part; every other part is byte-identical.

    `part` may also be "structure" or "harmony", which redraw those and
    clear the resolved values so they are generated afresh. A maker clicking
    re-roll wants fresh entropy (the default); a caller that must reproduce
    its output - the critic's revision loop - passes its own `rng`.
    """
    params = song.params.copy()
    params.reseed(part, rng)
    if part == "sub" and params.bass is not None:
        # A new bassline, not the same shape: a fixed-shape style always
        # moves to another kind; a seeded one half the time.
        r = rng or random.Random()
        cur = params.bass.style
        if cur not in SEEDED_BASS or r.random() < 0.5:
            pool = [(n, w) for n, w in bass_style_pool(profile_mod.get(params.profile)) if n != cur]
            if pool:
                names, weights = zip(*pool)
                params.bass = replace(params.bass, style=r.choices(names, weights)[0])
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
        # Any of the twelve: the tonic is folded into the sub register below,
        # so every key fits. Drawn from its own stream (the shared one still
        # takes its old draw, so nothing downstream moves).
        rng.choice((0, 1, 2, 3, 4))
        root = 33 + random.Random(f"{params.seed}:key").randrange(12)
        scale = Scale(root, rng.choice(prof.modes))
    # Anchor the tonic at the bottom of the sub register: the bassline is
    # built from degrees above the root and needs room to move up.
    root = pitch.transpose_into(scale.root, prof.sub.low, prof.sub.low + 11)
    scale = Scale(root, scale.name)
    params.key = f"{pitch.note_name(scale.root)} {scale.name}"
    if not params.bpm:
        lo, hi = prof.bpm_range if prof.bpm_range[1] else (prof.bpm, prof.bpm)
        params.bpm = random.Random(f"{params.seed}:bpm").randint(lo, hi)
    # Everything is written dead on the grid. Swing and feel are the
    # Deluge's job: its own swing setting moves every clip together, where
    # swing written into the notes fights it and drifts against it.
    params.swing = 0.0
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
        params.structure = structure.params_for_family(prof.family, srng, params.bpm or prof.bpm)
    arrangement = structure.generate(params.structure, srng, name=prof.family)
    if prof.bubble_through and prof.pad_style == "bubble":
        for sec in arrangement.sections:
            if "chords" in sec.active and "pad" not in sec.active:
                sec.active = tuple(sec.active) + ("pad",)
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
    if params.chord_size is None:
        # Dub mostly skanks on sevenths - voiced rootless, so the stab is the
        # third, fifth and seventh over the bass's root: the m7 colour of the
        # genre. Garage lives on the minor ninth - the Rhodes chord - and
        # dubstep, mostly hard plain triads, sometimes keeps garage's ninths.
        # Own stream, so it cannot move the harmony draws above.
        crng = random.Random(f"{params.seed}:chordsize")
        params.chord_size = crng.choice({"dub": (3, 4, 4), "garage": (4, 5, 5, 5)}.get(
            prof.family, (3, 3, 4, 5)))
    size = int(params.chord_size)

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
        progs, core = _harmony_for_sections(scale, sections, params, hrng,
                                            max_peak_chords=prof.drop_chords)

    # One chord plan per section, laid out at the section's absolute start,
    # with voice leading continuing across the seams so a progression change
    # moves by the smallest step instead of every voice jumping to home.
    plan_pieces: list[harmony.HarmonyPlan] = []
    voiced_pieces: list[harmony.HarmonyPlan] = []
    previous: chord_mod.Chord | None = None
    # The hook's params are resolved here, before the chords, because where
    # the hook sings the harmony is laid out to its phrase length.
    if params.hook is None:
        params.hook = hook_mod.params_for(prof.feel, random.Random(f"{params.seed}:hookparams"),
                                          family=prof.family)
    phrase_bars = params.hook.unit_bars * 4
    if params.dynamics is None:
        params.dynamics = dynamics.params_for(
            prof.family, random.Random(f"{params.seed}:dynamics"), prof.bass_drops)
    for i, (sec, prog) in enumerate(zip(sections, progs)):
        cycle = len(prog.degrees) * params.bars_per_chord
        sings = structure.kind_of(sec) in ("peak", "body") and sec.plays("lead") or \
            structure.kind_of(sec) == "peak"
        aligned = cycle > 0 and (abs(cycle % phrase_bars) < 1e-9 or abs(phrase_bars % cycle) < 1e-9)
        if sings and not aligned and len(prog.degrees) > 1:
            piece = harmony.realise_cycle(scale, prog, sec.bars, phrase_bars,
                                          size=size).offset(sec.start_tick)
        else:
            # A drop plays its whole cycle at least once: when the chords do
            # not fit, they move faster - in a developed later drop that is
            # the harmonic rhythm quickening, itself a development.
            bpc = params.bars_per_chord
            if sec.energy >= PEAK_ENERGY:
                while bpc > 1 and len(prog.degrees) * bpc > sec.bars:
                    bpc /= 2
            piece = harmony.realise(
                scale, prog, sec.bars, bpc, size=size,
            ).offset(sec.start_tick)
        nxt = sections[i + 1] if i + 1 < len(sections) else None
        if nxt is not None and dynamics.drop_entry(sec, nxt) and \
                random.Random(f"{params.seed}:cadence:{i}").random() < params.dynamics.cadence:
            piece = _approach(piece, sec, harmony.tension_chord(scale, size))
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

    # The key lift: the last drop of a song with two or more, up a semitone
    # or a tone, from the approach bars on (they become the new key's
    # dominant). Decided here, applied to the finished notes.
    lift = (0, 0)
    drops = [i for i in range(1, len(sections)) if dynamics.drop_entry(sections[i - 1], sections[i])]
    lrng = random.Random(f"{params.seed}:lift")
    if len(drops) >= 2 and lrng.random() < params.dynamics.lift:
        j = drops[-1]
        before = sections[j - 1]
        start = sections[j].start_tick
        if random.Random(f"{params.seed}:cadence:{j - 1}").random() < params.dynamics.cadence:
            start = before.start_tick + (before.bars - (2 if before.bars >= 8 else 1)) * TICKS_PER_BAR
        lift = (start, random.Random(f"{params.seed}:liftsize").choice((1, 2, 2)))

    # Drums: a feel varied into a specific spec.
    if params.drums is None:
        # The feel is drawn from the idiom's whole family, not just its first
        # member: one-drop, flyer, pickup and rockers are four records.
        style = random.Random(f"{params.seed}:feel").choice(prof.drum_styles)
        base = groove.FEELS.get(style, groove.FEELS.get(prof.drum_style, groove.HALF_TIME))
        params.drums = groove.vary(base, rng, amount=0.15)
    drum_rng = random.Random(params.seed_for("drums"))
    bar_a = groove.drums(params.drums, drum_rng)
    bar_b = groove.drums(groove.vary(params.drums, drum_rng, 0.1), drum_rng)
    # A one-drop drummer who plays rockers now and then: the riddim's own
    # snare, hats and rim, the kick on one and three (groove.ROCKERS's anchors).
    rockers = []
    if prof.name in ROCKERS_BLOCKS and 0 not in params.drums.kick_anchors:
        rspec = replace(params.drums, kick_anchors=groove.ROCKERS.kick_anchors,
                        kick_density=groove.ROCKERS.kick_density,
                        kick_syncopation=groove.ROCKERS.kick_syncopation)
        rrng = random.Random(f"{params.seed}:rockers")
        rockers = [{**bar, "kick": groove.drums(rspec, rrng)["kick"]} for bar in (bar_a, bar_b)]

    # Stabs.
    feel = prof.feel
    if params.stab_density is None:
        params.stab_density = round(rng.uniform(0.5, 1.0), 2)
    if params.stab_syncopation is None:
        params.stab_syncopation = round(rng.uniform(0.6, 0.95), 2)
    if params.stab_pattern is None:
        srng = random.Random(f"{params.seed}:stabpattern")
        named = [n for n in prof.stab_patterns if n in rhythm.STAB_PATTERNS.get(feel, {})]
        params.stab_pattern = srng.choice(named) if named and srng.random() < 0.6 else "generated"
    if params.stab_pattern in rhythm.STAB_PATTERNS.get(feel, {}):
        stab = rhythm.stab(feel, params.stab_pattern, velocity=92,
                           note_length=prof.skank_length)
    else:
        stab = groove.generate(
            groove.stab_spec(feel, params.stab_density, params.stab_syncopation,
                             note_length=prof.skank_length),
            random.Random(params.seed_for("chords")),
        )

    # Bass.
    if params.bass is None:
        params.bass = bassline.params_for_feel(feel, rng, energy=0.6)
        srng = random.Random(f"{params.seed}:bassstyle")
        names, weights = zip(*bass_style_pool(prof))
        params.bass = replace(params.bass, style=srng.choices(names, weights)[0])
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
    # Some songs play a rhythm real records play instead - drawn on its own
    # stream after the generated pair, so nothing else moves.
    mrng = random.Random(f"{params.seed}:minedbass")
    if mrng.random() < prof.mined_bass:
        cells = mined_cells.bass_pair(prof.name, mrng) or cells

    # Lead. The style is drawn from the shared stream (as it always was);
    # the performance dials from their own, so nothing downstream moves.
    if params.lead_style is None:
        params.lead_style = rng.choice(prof.lead_styles)
        # A sparse kind of lead delivered sparsely vanishes from the drop:
        # drones, stabs and slow song lines are sung as written, not at
        # half speed or with every other phrase rested. (Drawn only - a
        # style the maker chose is kept.)
        ph = params.hook.phrasing if params.hook else "call"
        if (ph in ("drone", "stabs") and params.lead_style in ("long_tones", "call_response")) or \
                (ph == "song" and params.lead_style == "long_tones"):
            params.lead_style = "hook"
    if params.lead is None:
        params.lead = lead_params_for(prof.family, params.lead_style,
                                      random.Random(f"{params.seed}:leadparams"),
                                      params.hook.phrasing if params.hook else "call")
    motif = melody.Motif([], STEPS_PER_BAR)

    # The riddim at full weight, for the peaks. Drawn after bars A and A' so
    # it cannot move them.
    peak_spec = groove.intensify(params.drums)
    drum_peak = [groove.drums(peak_spec, drum_rng),
                 groove.drums(groove.vary(peak_spec, drum_rng, 0.1), drum_rng)]
    # The last drop plays it harder again: the same riddim, more of it.
    last_spec = groove.intensify(peak_spec, 0.8)
    drum_peak += [groove.drums(last_spec, drum_rng),
                  groove.drums(groove.vary(last_spec, drum_rng, 0.1), drum_rng)]

    # The theme, how the band talks around it, and how tension is built.
    # Each is resolved from its own stream so none can shift another's.
    if params.interplay is None:
        params.interplay = interplay.params_for(
            prof.family, random.Random(f"{params.seed}:interplay"))
    if params.dynamics is None:
        params.dynamics = dynamics.params_for(
            prof.family, random.Random(f"{params.seed}:dynamics"), prof.bass_drops)
    # The stabs' real ceiling - the highest note any voicing plays, plus the
    # inversion a lifted stab can climb - not the top of their window: the
    # lead sits above what the chords actually play, not above where they
    # could in theory reach, which put every tune a fourth too high.
    tops = [max(ev.chord.pitches) for ev in voiced.events if ev.chord.pitches]
    chord_top = min(prof.chords.high, (max(tops) + 3) if tops else prof.chords.high)
    the_hook, lines, shift = _theme(params, prof, scale, plan, sections, chord_top)

    cell = the_hook.cell.tones if the_hook else []
    echo = tuple(t.step - cell[0].step for t in cell[:3]) if len(cell) >= 2 else (0.0, 3.0)

    return Material(
        scale=scale, plan=plan, voiced=voiced, motif=motif,
        drums=params.drums, drum_bars=[bar_a, bar_b], drum_rockers=rockers, stab=stab,
        bass=params.bass, bass_cells=cells, bass_pitch_seed=bass_rng.random(),
        lead_style=params.lead_style, lead=params.lead, swing=0.0,
        progressions=tuple(progs),
        hook=the_hook, lines=lines, interplay=params.interplay,
        lead_seed=params.seed_for("lead"),
        dynamics=params.dynamics, lead_shift=shift, drum_peak=drum_peak,
        echo=echo, sections=list(sections),
        silences=_silences(params, sections),
        mutes=_layers(params, sections),
        lift=lift,
    )


def lead_window(prof, lp) -> tuple[int, int]:
    """The lead's register window, moved by its octave dial: the dial moves
    the whole window, so turning it up is always heard - it is never folded
    back into the profile's default range."""
    o = 12 * int(round(lp.octave)) if lp is not None else 0
    return prof.lead.low + o, prof.lead.high + o


def lead_params_for(family: str, style: str, rng: random.Random,
                    phrasing: str = "call") -> melody.LeadParams:
    """Performance dials by idiom. A 140 lead is short, hard-accented and
    flicks octaves; a dub lead is legato, slides, grace-noted and echoed."""
    u = lambda lo, hi: round(rng.uniform(lo, hi), 2)  # noqa: E731
    if family == "garage":
        # Garage leads are chopped vocals and organ riffs: clipped, bouncy,
        # accented, with the odd octave flip and little slide.
        lp = melody.LeadParams(octave=rng.choice((0, 0, 0, 1)), gate=u(0.5, 0.8),
                               slide=u(0.05, 0.25), accent=u(0.5, 0.85), grace=u(0.0, 0.15),
                               flick=u(0.1, 0.3), echo=u(0.1, 0.35), ornament=u(0.0, 0.05))
    elif family == "dubstep":
        lp = melody.LeadParams(octave=rng.choice((0, 0, 0, 1)), gate=u(0.65, 0.9),
                               slide=u(0.15, 0.45), accent=u(0.4, 0.8), grace=u(0.05, 0.25),
                               flick=u(0.05, 0.25), echo=u(0.0, 0.2), ornament=u(0.0, 0.08))
    else:
        lp = melody.LeadParams(octave=0, gate=u(0.8, 1.0),
                               slide=u(0.25, 0.6), accent=u(0.25, 0.55), grace=u(0.1, 0.35),
                               flick=u(0.0, 0.08), echo=u(0.15, 0.45), ornament=u(0.02, 0.15))
    # Each kind of lead is played its own way - this is half of why two
    # leads sound like two different instruments.
    if phrasing == "stabs":
        lp = replace(lp, gate=u(0.12, 0.35), slide=0.0, accent=u(0.7, 1.0),
                     flick=u(0.2, 0.5), echo=u(0.2, 0.5), ornament=0.0,
                     octave=rng.choice((0, 0, 1)))
    elif phrasing == "riff":
        lp = replace(lp, gate=u(0.3, 0.6), slide=u(0.0, 0.2), accent=u(0.6, 0.9),
                     grace=u(0.0, 0.15), ornament=0.0)
    elif phrasing == "chop":
        # A vocal chop: clipped syllables, hard accents, no slide - the
        # sampler cutting the phrase, not a singer carrying it.
        lp = replace(lp, gate=u(0.4, 0.7), slide=0.0, accent=u(0.6, 0.95),
                     grace=0.0, flick=u(0.05, 0.25), echo=u(0.15, 0.45), ornament=0.0,
                     octave=rng.choice((0, 0, 1)))
    elif phrasing == "arp":
        lp = replace(lp, gate=u(0.35, 0.7), slide=0.0, accent=u(0.4, 0.8),
                     grace=0.0, flick=0.0, echo=u(0.1, 0.4), ornament=0.0,
                     octave=rng.choice((0, 0, 1)))
    elif phrasing in ("song", "drone"):
        lp = replace(lp, gate=u(0.85, 1.0), slide=u(0.3, 0.7), accent=u(0.2, 0.5),
                     grace=u(0.15, 0.4), flick=0.0,
                     ornament=u(0.05, 0.25) if phrasing == "drone" else lp.ornament,
                     octave=rng.choice((0, 0, 1)) if phrasing == "song" else 0)
    if style == "long_tones":
        lp = replace(lp, gate=max(lp.gate, 0.9), flick=0.0)
    return lp


def _approach(piece: harmony.HarmonyPlan, sec: Section, chord) -> harmony.HarmonyPlan:
    """The last bars of a section that runs into a drop move to a tension
    chord, so the drop's downbeat is a resolution as well as an impact."""
    bars = 2 if sec.bars >= 8 else 1
    cut = sec.start_tick + (sec.bars - bars) * TICKS_PER_BAR
    events = []
    for ev in piece.events:
        if ev.start >= cut:
            continue
        if ev.end > cut:
            ev = chord_mod.ChordEvent(ev.chord, ev.start, cut - ev.start, ev.label, dict(ev.meta))
        events.append(ev)
    events.append(chord_mod.ChordEvent(chord, cut, bars * TICKS_PER_BAR, chord.symbol(),
                                       {"approach": True}))
    return harmony.HarmonyPlan(piece.scale, piece.progression, events)


def _hook_anchor(sections: list[Section]) -> Section:
    """Where the hook is written against: the first peak (that is where it
    has to be strongest), else the first section the lead sings in."""
    for s in sections:
        if structure.kind_of(s) == "peak":
            return s
    for s in sections:
        if s.plays("lead"):
            return s
    return sections[0]


def _chords_from(plan: harmony.HarmonyPlan, start_tick: int, offset: float = 0.0):
    def chord_at(step: float):
        ev = plan.at(start_tick + int(round((offset + step) * TICKS_PER_16TH)))
        return ev.chord if ev is not None else None
    return chord_at


def _theme(params: SongParams, prof: Profile, scale: Scale,
           plan: harmony.HarmonyPlan, sections: list[Section], chord_top: int | None = None):
    """Write the hook against the first drop's chords, place it in the lead's
    register, and plan its treatment in every section.

    The hook has its own seed: re-rolling it is a new tune over the same
    band. The plan is drawn from that seed too, never from a part's stream,
    so the parts that answer the hook move with it and nothing else does.
    """
    hp = params.hook
    hrng = random.Random(params.seed_for("hook"))
    anchor = _hook_anchor(sections)
    the_hook = hook_mod.best(hp, scale, hrng, _chords_from(plan, anchor.start_tick),
                             hp.unit_bars * 16)
    # The B-theme: written against the same chords, from its own stream, so
    # it never moves the hook.
    hook_b = hook_mod.best(hook_mod.contrast_params(hp), scale,
                           random.Random(f"{params.seed_for('hook')}:b"),
                           _chords_from(plan, anchor.start_tick), hp.unit_bars * 16)
    # A second theme starts somewhere the first does not: if both open on
    # the same note, B moves up a third (and is refitted to the chords when
    # it is realised), so it is heard as new from its first note.
    a0 = next((t.degree for u in the_hook.units for t in u.tones), None)
    b0 = next((t.degree for u in hook_b.units for t in u.tones), None)
    if a0 is not None and b0 is not None and (a0 - b0) % scale.size == 0:
        hook_b = hook_mod.Hook(
            [hook_mod.Unit(u.role, [replace(t, degree=t.degree + 2) for t in u.tones], u.steps)
             for u in hook_b.units],
            hook_b.params, hook_b.unit_steps, hook_b.contour, hook_b.score, hook_b.report)

    # One register for the whole song, set by the hook as written: its mean
    # sits just above the chord window, so the tune is never inside the
    # stabs. Moving the phrase as a block keeps its leap - folding note by
    # note would flatten the very interval people sing back.
    tones = hook_mod.realise(the_hook, scale, _chords_from(plan, anchor.start_tick))
    pitches = [t.midi(scale) for t in tones] or [scale.root + 36]
    mean = sum(pitches) / len(pitches)
    # Of the octave placements, the one that keeps the most of the tune out
    # of the stabs (above the chord window) and inside the lead's window -
    # judged note by note, not by the mean, so a wide hook's low end does
    # not sink into the chords. Ties go low: a lead should not screech.
    lo, hi = lead_window(prof, params.lead)
    floor = (chord_top if chord_top is not None else prof.chords.high) \
        + 12 * int(round(params.lead.octave))
    base = 12 * round((floor + 6 - mean) / 12)

    # A few low notes may dip beside the stabs - they are short and the tune
    # crosses them for a moment - but a tune living far above the chords
    # screeches: notes more than an octave and a step over the stabs' top
    # count against a placement as much as notes inside them.
    ceiling = floor + 14

    def cost(sh):
        inside = sum(1 for x in pitches if x + sh <= floor)
        tolerated = max(0, inside - len(pitches) // 4)
        high = sum(1 for x in pitches if x + sh > ceiling)
        over = sum(1 for x in pitches if x + sh > hi)
        under = sum(1 for x in pitches if x + sh < lo)
        return (tolerated + high + 2 * over + 2 * under, sh)
    shift = min((base - 12, base, base + 12), key=cost)

    def fits_lift(ts) -> bool:
        return bool(ts) and max(t.midi(scale) for t in ts) + shift + 12 <= hi

    lines: dict[int, Line] = {}
    peaks = 0
    for i, sec in enumerate(sections):
        kind = structure.kind_of(sec)

        def realise_at(h, off, sec=sec):
            fitted = hook_mod.realise(h, scale, _chords_from(plan, sec.start_tick, off))
            return [replace(t, step=t.step + off) for t in fitted]

        line = interplay.plan(
            kind, sec.bars, the_hook, realise_at, params.lead_style or "hook",
            params.interplay, peaks, random.Random(f"{params.seed_for('hook')}:{i}"),
            lead_plays=sec.plays("lead"), fits_lift=fits_lift,
            roles=tuple(r for r in ("sub", "chords") if sec.plays(r)),
            hook_b=hook_b,
        )
        if kind in ("build", "intro", "outro"):
            line.tones = hook_mod.fit(line.tones, scale, _chords_from(plan, sec.start_tick))
        lines[sec.start_bar] = line
        if kind == "peak":
            peaks += 1
    return the_hook, lines, shift


HAT, SHAKER = 38, 41
# Profiles whose one-drop drummer slips into rockers: every other four-bar
# block of a section that is not a peak plays the riddim with the kick on one
# and three. Roots records' four bars repeat far less than ours (drum_feel
# 0.54 against 0.72); on the goal panel (8 seeds, against 6349062) this moved
# ours to 0.69, FAD 0.341 -> 0.336, closeness 0.820 -> 0.822, real 0.92 ->
# 0.91. (Rockers on the every-fourth-bar variant alone did less.)
# Off by ear: Ed preferred roots without them in the blind A/B (the judge's
# gain was FAD 0.341 -> 0.336, inside its noise). The machinery stays.
ROCKERS_BLOCKS: set[str] = set()


def _layers(params: SongParams, sections: list[Section]) -> list:
    """How a long drop unfolds, eight bars at a time.

    The first block holds the hats and shaker back - the kit arriving in
    full at bar nine is a lift of its own. In a thirty-two the third block
    strips to drums, bass and the tune (chords and pad out) before the last
    block brings everything back. Kick, snare, bass and lead always play:
    the drop never stops being the drop.
    """
    out = []
    low_end_out = params.structure is not None and params.structure.low_end_out
    for i, sec in enumerate(sections):
        kind = structure.kind_of(sec)
        if low_end_out and (kind in ("intro", "breakdown") or (i == 0 and kind == "build")):
            # The low end is held for the drop: the kit plays, its kick
            # does not.
            out.append((sec.start_tick, sec.start_tick + sec.length_ticks,
                        {("drums", KIT_ROWS["kick"])}))
            continue
        if kind == "intro" and sec.bars >= 8 and i == 0:
            # The beat lands halfway: hats, pad and bass set the scene
            # first, kick and snare arrive at the half - a layer, not a wall.
            half = (sec.bars // 2) * TICKS_PER_BAR
            out.append((sec.start_tick, sec.start_tick + half,
                        {("drums", KIT_ROWS["kick"]), ("drums", KIT_ROWS["snare"]),
                         ("drums", KIT_ROWS["rim"])}))
            continue
        if kind == "outro" and sec.bars >= 8 and i == len(sections) - 1:
            # Strip to the foundation: hats and shaker go for the last
            # block; a long outro loses its stabs a block before that.
            end = sec.start_tick + sec.length_ticks
            block = 4 * TICKS_PER_BAR if sec.bars < 16 else 8 * TICKS_PER_BAR
            out.append((end - block, end, {("drums", HAT), ("perc", None)}))
            if sec.bars >= 16:
                out.append((end - 2 * block, end, {("chords", None)}))
            continue
        if kind in ("peak", "body", "release") and sec.plays("sub"):
            drng = random.Random(f"{params.seed}:bassdrop:{i}")
            for a, b in dynamics.bass_drop_windows(sec, params.dynamics, drng):
                out.append((a, b, {("sub", None)}))
        if kind != "peak" or sec.bars < 16:
            continue
        if random.Random(f"{params.seed}:layers:{i}").random() >= params.dynamics.layering:
            continue
        # The last of several drops arrives whole: its first block holds
        # nothing back (it may still strip and return in its third).
        last_drop = sum(1 for s in sections if structure.kind_of(s) == "peak") >= 2 and \
            not any(structure.kind_of(s) == "peak" for s in sections[i + 1:])
        block = 8 * TICKS_PER_BAR
        if not last_drop:
            out.append((sec.start_tick, sec.start_tick + block,
                        {("drums", HAT), ("perc", None)}))
        if sec.bars >= 32:
            out.append((sec.start_tick + 2 * block, sec.start_tick + 3 * block,
                        {("chords", None), ("pad", None)}))
    return out


# A pull takes the room's low end, not only the sub's notes: on the desk the
# engineer's low cut thins the kick too. Measured on the renders, muting the
# sub alone left 25-70 Hz only 2-3 dB down (our 140 kick carries it) where the
# records drop 12 dB, so the kick row is high-passed for the pulled bar.
PULL_KICK_HPF = 0.35


def _thin_kick_in_pulls(part: Part, mutes) -> None:
    for a, b, who in mutes:
        if who == {("sub", None)}:
            part.automation.append(Automation("hpf", a, b - a, PULL_KICK_HPF, PULL_KICK_HPF,
                                              row=KIT_ROWS["kick"]))


def _apply_mutes(part: Part, mutes) -> None:
    if not mutes:
        return
    kept = []
    for n in part.notes:
        muted = any(a <= n.start < b and ((part.role, None) in who or (part.role, n.midi) in who)
                    for a, b, who in mutes)
        if not muted:
            kept.append(n)
    part.notes = kept


def _silences(params: SongParams, sections: list[Section]) -> list:
    """The windows where the band stops: the gap before each drop (everyone)
    and the switch-ups inside long peaks (everyone but the lead)."""
    dyn = params.dynamics
    out = []
    gap = dynamics.gap_ticks(dyn)
    for prev, sec in zip(sections, sections[1:]):
        if gap and dynamics.drop_entry(prev, sec):
            out.append((sec.start_tick - gap, sec.start_tick, ()))
    for i, sec in enumerate(sections):
        for a, b in dynamics.switch_windows(sec, dyn, random.Random(f"{params.seed}:switch:{i}")):
            out.append((a, b, ("lead",)))
    return out


# The energy boundaries that give a section its own harmonic personality:
# at or above this a section is a peak and earns a cadence; at or below it a
# hole and earns a rest chord; between, it sits on the tune's core.
PEAK_ENERGY = 0.8
HOLE_ENERGY = 0.35


def _progression_by_name(name: str, scale: Scale, lenient: bool = False) -> harmony.Progression:
    if name in harmony.BY_NAME:
        return harmony.BY_NAME[name]
    try:
        return _parse_roman(name, scale)
    except ValueError:
        if not lenient:
            raise
        return _parse_roman_by_degree(name, scale)


_NUMERALS = ("I", "II", "III", "IV", "V", "VI", "VII")


def _parse_roman_by_degree(text: str, scale: Scale) -> harmony.Progression:
    """A stored progression read in a key that spells it differently (the
    song changed mode since): each numeral by its scale degree, with the
    chord this mode builds there - never a diminished one, which a vamp
    cannot sit on (that degree falls back to the one before)."""
    degrees = []
    for token in (n.strip() for n in text.replace(" ", "-").split("-") if n.strip()):
        core = token.lstrip("b#").rstrip("°+7").upper()
        d = _NUMERALS.index(core) if core in _NUMERALS else 0
        if chord_mod.triad(scale, d).quality in ("dim", "aug"):
            d = degrees[-1] if degrees else 0
        degrees.append(d)
    return harmony.Progression(text, tuple(degrees or [0]), (scale.name,), 0.5, "re-read")


def _harmony_for_sections(
    scale: Scale,
    sections: list[Section],
    params: SongParams,
    hrng: random.Random,
    max_peak_chords: int | None = None,
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
            progs.append(_progression_by_name(spec.progression, scale, lenient=True))
            continue
        if sec.energy >= PEAK_ENERGY:
            if peak is None:
                peak = harmony.choose(scale.name, 0.85, hrng, tolerance=0.4,
                                      max_chords=max_peak_chords)
                progs.append(peak)
            else:
                # A later drop returns to the first one's loop, then develops
                # it - the return is recognisable, and it is not a repeat.
                # Own stream: nothing else in the harmony moves.
                peaks_so_far = sum(1 for s in sections[:i] if s.energy >= PEAK_ENERGY)
                progs.append(harmony.develop(
                    peak, scale, random.Random(f"{params.seed}:develop:{peaks_so_far}")))
        elif sec.energy <= HOLE_ENERGY:
            if release is None:
                release = harmony.choose(scale.name, 0.15, hrng)
            if 0 < i < len(sections) - 1 and _is_bridge(sec):
                # A breakdown in the middle of the song is its contrast - the
                # bridge - and goes somewhere the drop and the riddim never
                # do. Intro and outro keep the rest chord, so the song still
                # opens and closes on the same ground. Own stream.
                avoid = {p.name for p in progs if p is not None} | {release.name}
                brng = random.Random(f"{params.seed}:bridge")
                pool = [p for p in harmony.for_mode(scale.name) if p.name not in avoid
                        and len(p.degrees) >= 2 and 0.2 <= p.tension <= 0.65]
                progs.append(brng.choice(pool) if pool else release)
            else:
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


def _is_bridge(sec: Section) -> bool:
    """A mid-song breakdown (not a dub-out, which strips the riddim rather
    than leaving it)."""
    return sec.name.startswith("breakdown")


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
    dyn = m.dynamics
    prev, nxt = m.neighbours(section)
    into_drop = nxt is not None and dynamics.drop_entry(section, nxt)
    is_drop = prev is not None and dynamics.drop_entry(prev, section)
    gap = dynamics.gap_ticks(dyn) if into_drop else 0
    gap_steps = int(round(gap / TICKS_PER_16TH))
    roll_n = dynamics.roll_bars(dyn, section) if into_drop else 0
    peak = structure.kind_of(section) == "peak"
    sub = m.built.get("sub") if section.plays("sub") else None
    sub_cell = _bass_cell(sub, section) if sub is not None else []
    # The lead's onsets, section-relative, so the kit can make room for it.
    lead_on = m.line(section).onsets() if section.plays("lead") else set()
    # Each voice draws from its own stream per bar, so the kick following a
    # new bassline cannot nudge a hat's velocity.
    voice_seed = rng.random()
    peaks_before = [s for s in m.sections
                    if s.start_bar < section.start_bar and structure.kind_of(s) == "peak"]
    later_peak = peak and bool(peaks_before)
    switch_ends = {b for a, b, _ in m.silences if section.start_tick < b <= section.start_tick
                   + section.length_ticks and b - a < TICKS_PER_BAR}

    for bar in range(section.bars):
        bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
        # A ramped section climbs bar by bar; a flat one sits on its level.
        energy = structure.bar_energy(section, bar)
        density = structure.density_for(energy, "drums")
        vel_scale = structure.velocity_for(energy) / 100.0
        # A two-bar riddim: A A A A' - the variant on every fourth bar. A
        # peak plays the same riddim at full weight.
        bank = m.drum_bars
        if peak and m.drum_peak:
            bank = m.drum_peak[2:4] if (later_peak and len(m.drum_peak) >= 4) else m.drum_peak[:2]
        # A one-drop drummer's rockers block: its kick is the rockers anchors,
        # so it is not locked to the one-drop's.
        rock = bool(m.drum_rockers) and not peak and (bar // 4) % 2 == 1
        if rock:
            bank = m.drum_rockers
        voices = bank[1] if (bar % 4 == 3) else bank[0]
        bars_left = section.bars - bar
        rolling = bars_left <= roll_n
        ends_phrase = (bar + 1) % 16 == 0
        ends_section = bar == section.bars - 1
        # The last bar of a section is a fill: the snare and rim let go of
        # the riddim and run into the next section's downbeat. Inside a
        # section, an eight-bar phrase end gets a fill by `fills`, otherwise
        # the ordinary pattern thickened. A build into a drop rolls instead.
        fill_this_bar = ends_section and section.bars >= 4 and not rolling
        # A fill never talks over the lead: if the tune is singing through the
        # back half of this bar, the phrase fill sits out and a section-end
        # fill plays under it, softly.
        lo = bar * STEPS_PER_BAR
        lead_late = any(lo + 8 <= st < lo + 16 for st in lead_on)
        phrase_fill = (ends_phrase and not ends_section and not rolling
                       and not lead_late and rng.random() < dyn.fills)

        for voice, pattern in voices.items():
            if voice not in KIT_ROWS or not pattern.hits:
                continue
            if (fill_this_bar or phrase_fill) and voice in ("snare", "rim"):
                continue  # the fill owns this bar
            if rolling and voice in ("snare", "rim"):
                continue  # the roll owns these
            if voice == "kick" and sub is not None and not rock:
                pattern = _lock_kick(pattern, sub_cell, m)
            # One stream per voice for the whole section: at a given density
            # the same hits survive every bar, so the groove repeats instead of
            # reshuffling - a hat that changes every bar sounds out of time.
            vrng = random.Random(f"{voice_seed}:{voice}")
            keep = density if voice in ("hat", "rim") else min(1.0, density + 0.35)
            climb = 0
            if voice == "hat" and into_drop and section.ramp > 0 and section.bars >= 8 \
                    and 2 <= bars_left <= 4:
                # The hats climb the ladder over the last bars of a build -
                # a rung, then two - and play in full: the top end tightens
                # with the roll, and the drop's downbeat releases it.
                climb = 1 if bars_left > 2 else 2
                pattern, keep = groove.hat_pattern(m.drums, climb), 1.0
            if voice == "hat" and not climb and prof.drummer_blocks:
                pattern = _hat_block(pattern, m, prof, voice_seed, bar)
                if pattern is None:
                    continue
            p = rhythm.thin(pattern, keep, vrng)
            if rolling and bars_left == 1 and voice in ("kick", "hat"):
                # The floor goes out from under the last bar of the roll.
                p = rhythm.mask(p, lambda st: st < 8)
            # On the grid: velocity is humanised, timing never is.
            p = rhythm.humanize(p, 0.0, 6, vrng)
            _emit(part, [(KIT_ROWS[voice], h.step, min(h.length, 0.9), h.velocity * vel_scale)
                         for h in p.sorted_hits()], bar_tick)

        if prof.drummer and bar % 4 != 3 and not (rolling or fill_this_bar or phrase_fill):
            # Its own stream per bar, so a song without a drummer is note
            # for note what it was.
            drng = random.Random(f"{voice_seed}:drummer:{bar}")
            if drng.random() < prof.drummer:
                _drummer_touch(part, voices, bar_tick, vel_scale, drng)

        if rolling:
            _emit(part, [(KIT_ROWS["snare"], s, 0.5, v)
                         for s, v in dynamics.roll(bars_left, roll_n, dyn, rng,
                                                   gap_steps if bars_left == 1 else 0)],
                  bar_tick)
        if fill_this_bar or phrase_fill:
            fill_energy = energy if fill_this_bar else min(energy, 0.5)
            under = 0.75 if lead_late else 1.0
            for voice, pattern in fill.make(fill_energy, rng).items():
                if voice not in KIT_ROWS:
                    continue
                _emit(part, [(KIT_ROWS[voice], h.step, min(h.length, 0.9),
                              h.velocity * vel_scale * under) for h in pattern.sorted_hits()],
                      bar_tick)

        # Impacts: the drop's downbeat, every eighth bar of a peak, and the
        # bar after a switch-up - the band coming back in.
        if bar == 0 and is_drop and dyn.impact > 0.05:
            _emit(part, [(KIT_ROWS["crash"], 0, 8, 96 + 30 * dyn.impact)], bar_tick)
        elif peak and bar % 8 == 0 and bar > 0 and rng.random() < dyn.impact:
            _emit(part, [(KIT_ROWS["crash"], 0, 8, 72 + 20 * dyn.impact)], bar_tick)
        elif bar_tick in switch_ends:
            _emit(part, [(KIT_ROWS["crash"], 0, 8, 90)], bar_tick)

    # The siren: the sound system's own voice, fired into a dub-out (a bar
    # after the band drops away) and on the downbeat of a later drop - never
    # the first, which the band alone has to land. Its own stream, so a song
    # without one is note for note what it was.
    if prof.sirens > 0:
        srng = random.Random(f"{voice_seed}:siren")
        if (section.dub_throw or (is_drop and peaks_before)) and srng.random() < prof.sirens:
            at = section.start_tick + (TICKS_PER_BAR if section.dub_throw and section.bars > 2 else 0)
            part.add(Note(KIT_ROWS["siren"], at, 2 * TICKS_PER_BAR, 100))

    if into_drop:
        span = dynamics.riser_span(section, dyn, gap)
        if span is not None:
            start, length = span
            part.add(Note(KIT_ROWS["riser"], start, length, int(80 + 40 * dyn.riser)))


def _hat_block(pattern, m: Material, prof: Profile, voice_seed: float, bar: int):
    """The hat figure for this bar's eight-bar block: the riddim's, or - by
    `drummer_blocks` - a rung busier, only the offbeat eighths, or nothing
    for the block's first four bars (None). Its own stream per block."""
    brng = random.Random(f"{voice_seed}:hatblock:{bar // 8}")
    if brng.random() >= prof.drummer_blocks:
        return pattern
    kind = brng.choice(("busier", "offbeats", "out"))
    if kind == "busier":
        return groove.hat_pattern(m.drums, 1)
    if kind == "offbeats":
        return rhythm.mask(pattern, lambda st: st % 4 == 2) if any(
            h.step % 4 == 2 for h in pattern.hits) else pattern
    return None if bar % 8 < 4 else pattern


def _drummer_touch(part: Part, voices: dict, bar_tick: int, vel_scale: float, rng) -> None:
    """One thing a player adds to a bar of the riddim: ghost snares between
    the hits, a closed hat opened on an "and", or a rim pickup into the next
    bar - only where the pattern leaves the step free."""
    taken = {v: {h.step for h in p.hits} for v, p in voices.items()}
    snare = taken.get("snare", set()) | taken.get("rim", set())
    kind = rng.choice(("ghost", "ghost", "open", "rim"))
    if kind == "ghost":
        free = [s for s in (6, 10, 13, 15) if s not in snare]
        notes = [(KIT_ROWS["snare"], s, 0.5, rng.uniform(26, 40))
                 for s in rng.sample(free, min(len(free), rng.choice((1, 2))))]
    elif kind == "open":
        step = rng.choice((6, 14))
        at = bar_tick + step * TICKS_PER_16TH
        part.notes = [n for n in part.notes if not (n.midi == KIT_ROWS["hat"] and n.start == at)]
        notes = [(KIT_ROWS["hat"], step, 2.0, rng.uniform(66, 82))]
    else:
        free = [s for s in (7, 15) if s not in snare]
        notes = [(KIT_ROWS["rim"], s, 0.5, rng.uniform(48, 62)) for s in free[-1:]]
    _emit(part, [(m, st, ln, v * vel_scale) for m, st, ln, v in notes], bar_tick)


def _bass_cell(sub: Part, section: Section) -> list[tuple[int, float]]:
    """The bass's usual positions in a section: each step with how often the
    bass starts a note there across the section's bars. The kick locks to
    this, not to each bar's notes - so it plays one steady pattern with the
    bass instead of chasing every reply and fill."""
    counts: dict[int, int] = {}
    for n in sub.notes:
        if section.start_tick <= n.start < section.start_tick + section.length_ticks:
            st = (n.start % TICKS_PER_BAR) // TICKS_PER_16TH
            counts[st] = counts.get(st, 0) + 1
    return sorted(((st, c / section.bars) for st, c in counts.items()), key=lambda x: x[0])


def spec_is_straight(m: Material) -> bool:
    return m.drums.feel != "half_time"


def _lock_kick(pattern, cell: list[tuple[int, float]], m: Material):
    """The kick plays with the bass. The genre's anchors stay - the one-drop
    on 3, half-time on 1, steppers on every beat - and every other kick in
    the bar moves onto a bass note: the strongest ones first, never onto the
    backbeat, never crowding an anchor. Kick and sub become one instrument
    instead of two parts that happen to share a tempo."""
    anchors = set(m.drums.kick_anchors)
    backbeat = set(m.drums.backbeat)
    extra = [h for h in pattern.hits if int(round(h.step)) not in anchors]
    if not extra or not cell:
        return pattern
    cands = []
    one_drop = 0 not in anchors and 8 in anchors and spec_is_straight(m)
    two_step = getattr(m.drums, "idiom", "roots") == "garage" and 4 not in anchors
    for st, share in cell:
        if share < 0.4:
            continue            # only where the bass plays in most bars
        if one_drop and st == 0:
            continue            # the one-drop's empty one stays empty
        if two_step and st % 4 == 0:
            continue            # 2-step leaves the beats it skips empty
        if st in backbeat or st in anchors or any(abs(st - a) < 2 for a in anchors):
            continue
        cands.append((share * 4 + metric_weight(st), st))
    chosen = sorted({st for _, st in sorted(cands, reverse=True)[:len(extra)]})
    if len(chosen) < len(extra):
        # A held or pulsing bass leaves the kick little to lock to: the rest
        # of its extra hits keep their own eighth-note places.
        for h in sorted(extra, key=lambda h: -metric_weight(int(h.step))):
            st = int(round(h.step)) // 2 * 2
            if len(chosen) >= len(extra):
                break
            if st not in chosen and st not in backbeat and st not in anchors \
                    and not any(abs(st - a) < 2 for a in anchors) \
                    and not (two_step and st % 4 == 0):
                chosen.append(st)
        chosen = sorted(chosen)
    vel = int(sum(h.velocity for h in extra) / len(extra))
    hits = [h for h in pattern.hits if int(round(h.step)) in anchors]
    hits += [rhythm.Hit(float(st), vel, 0.9) for st in chosen]
    return rhythm.Pattern(pattern.length, sorted(hits, key=lambda h: h.step),
                          pattern.steps_per_bar, pattern.name)


def _build_perc(part, section, prof, m: Material, rng):
    # One figure for the whole section, on the offbeats: a shaker that moves
    # onto the beat every other bar sounds like it is losing time.
    for bar in range(0, section.bars, 2):
        energy = structure.bar_energy(section, bar)
        density = structure.density_for(energy, "perc")
        # Offbeat eighths, filling in sixteenths only when the section is full.
        steps = [2, 6, 10, 14] + ([3, 7, 11, 15] if density > 0.85 else [])
        pattern = rhythm.from_steps(steps, STEPS_PER_BAR)
        pattern = rhythm.accent(pattern, strong=88, weak=58)
        if getattr(m.drums, "idiom", "roots") == "garage" and spec_is_straight(m) and density > 0.6:
            # The garage shaker runs in sixteenths under the skipping hats,
            # leaning on the offbeat - the engine the Deluge's swing turns
            # into the shuffle.
            pattern = rhythm.Pattern(STEPS_PER_BAR, [
                rhythm.Hit(float(s), 82 if s % 4 == 2 else (40 if s % 4 == 0 else 54), 0.5)
                for s in range(STEPS_PER_BAR)], STEPS_PER_BAR)
        two_bar = rhythm.every_other_bar(pattern, pattern, bars=2)
        bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
        shaken = rhythm.humanize(two_bar, 0.0, 8, rng)
        vel_scale = structure.velocity_for(energy) / 100.0
        _emit(part, [(KIT_ROWS["shaker"], h.step, 0.5, h.velocity * vel_scale)
                     for h in shaken.sorted_hits()], bar_tick)
    _build_hands(part, section, prof, m)


def _build_hands(part, section, prof, m: Material):
    """The hand percussion over the shaker: one two-bar part per section,
    repeated as a player repeats it, the last bar of every eight a call
    (a tom run, or the high conga picking up into the one). Its own stream
    per section, so a section re-rolls alone."""
    if m.hands is None:
        m.hands = percussion.params_for(prof.family, random.Random(f"{m.seeds.get('perc', 0)}:hands"))
    hrng = random.Random(f"{m.seeds.get('perc', 0)}:hands:{section.start_bar}")
    energy = structure.bar_energy(section, max(0, section.bars - 1))
    two = percussion.pattern(m.hands, prof.beat_steps, energy, hrng)
    vel_scale = structure.velocity_for(energy) / 100.0
    for bar in range(0, section.bars, 2):
        notes = []
        for voice, hits in two.items():
            for step, vel in hits:
                b = bar + int(step // STEPS_PER_BAR)
                if b >= section.bars:
                    continue
                if (b + 1) % 8 == 0 and step % STEPS_PER_BAR >= 8:
                    continue          # the call owns the end of the phrase
                notes.append((KIT_ROWS[voice], step, 0.5, vel * vel_scale))
        _emit(part, notes, (section.start_bar + bar) * TICKS_PER_BAR)
    for bar in range(7, section.bars, 8):
        fill_ = percussion.call(m.hands, random.Random(f"{m.seeds.get('perc', 0)}:call:{section.start_bar}:{bar}"))
        notes = [(KIT_ROWS[v], s, 0.5, vel * vel_scale) for v, hits in fill_.items() for s, vel in hits]
        _emit(part, notes, (section.start_bar + bar) * TICKS_PER_BAR)


def _build_sub(part, section, prof, m: Material, rng):
    """The bassline: generated phrases, cell restated and answered - and
    placed around the hook.

    The cell rhythm is fixed for the song (it is the bassline's identity)
    and its pitch choices are replayed from a fixed seed, so every
    restatement is a restatement. Answer bars draw fresh randomness. Peaks
    use the second, busier cell; a 32-bar peak switches cell halfway.

    Then the conversation: where the theme plan hands the sub the hook (a
    pass the lead sits out, or a unison pass), the sub plays the hook's
    rhythm on the chord root - the bass *is* the riff. Where the lead
    breathes, the sub may answer with a pickup into the next downbeat.
    """
    params = m.bass
    density = structure.density_for(section.energy, "sub")
    phrase_len = params.phrase_bars
    is_peak = section.energy >= 0.85
    line = m.line(section)
    ip = m.interplay
    busy = _lead_busy_beats(m, section)
    fill_params = replace(
        params, density=min(0.6, params.density + params.fill_density_boost),
        syncopation=min(1.0, params.syncopation + 0.15),
        approach=min(1.0, params.approach + 0.2),
    )

    for start in range(0, section.bars, phrase_len):
        bars = min(phrase_len, section.bars - start)
        roots = []
        for b in range(bars + 1):
            ev = m.plan.at((section.start_bar + start + b) * TICKS_PER_BAR)
            roots.append(ev.chord.degree if ev else 0)

        cell = m.bass_cells[1] if is_peak else m.bass_cells[0]
        if is_peak and section.bars >= 32 and start >= section.bars // 2:
            cell = m.bass_cells[0]
        fill_rhythm = groove.generate(fill_params.groove(), rng)

        motifs = []
        for b in range(bars):
            root, nxt = roots[b], roots[b + 1]
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
                    colour_params, roots[b + 1])
                if len(pcs(motifs)) >= 3:
                    break

        for b, motif in enumerate(motifs):
            rel_bar = start + b
            abs_bar = section.start_bar + rel_bar
            vel_energy = structure.bar_energy(section, rel_bar)
            lo_step = rel_bar * STEPS_PER_BAR
            hook_here = [t for t in line.bass_line if lo_step <= t.step < lo_step + STEPS_PER_BAR]
            if hook_here:
                notes = _bass_from_hook(hook_here, lo_step, roots[b], m, prof, section)
            else:
                if density < 0.85:
                    motif = melody.thin_tones(motif, density, rng)
                notes = bassline.constrain_register(motif, m.scale, prof.sub.low, prof.sub.high)
                notes = bassline.monophonic(notes)
                notes = [(mi, s, l * prof.bass_note_length, v) for mi, s, l, v in notes]
                notes = _bass_style(notes, m, prof, roots[b], roots[b + 1], rel_bar, rng)
                notes = _answer_breaths(notes, line, lo_step, roots[b + 1], m, prof, rng)
            if not hook_here:
                notes = _yield_to_lead(notes, busy, lo_step, m.interplay.space, m.bass.gate)
            replies = line.response_in(lo_step, lo_step + STEPS_PER_BAR, "sub")
            if replies:
                notes = _sub_reply(notes, replies, lo_step, roots[b], m, prof)
            notes = _on_eighths(notes, mined_cells.sixteenths(prof.name)
                                if prof.sub_sixteenths else frozenset())
            if not params.anchor_root and not hook_here:
                # A bass that skips the one still states a new chord: where
                # the harmony changes, its root lands with the one-drop kick
                # on beat 3 if nothing has said it by then.
                here = m.voiced.at(abs_bar * TICKS_PER_BAR)
                before = m.voiced.at(abs_bar * TICKS_PER_BAR - 1) if abs_bar > 0 else None
                changed = here is not None and (before is None or before.chord.degree != here.chord.degree)
                if changed and not any(st <= 8 for _, st, _, _ in notes):
                    r = _sub_note(m, prof, roots[b], roots[b])
                    nxt_st = min([st for _, st, _, _ in notes if st > 8] + [16.0])
                    notes = sorted(notes + [(r, 8.0, max(1.0, (nxt_st - 8) * m.bass.gate), 108)],
                                   key=lambda n: n[1])
            if prof.family == "dubstep" and structure.kind_of(section) == "peak":
                notes = _sub_economy(notes, m, prof, roots[b], roots[b + 1], rel_bar)
            notes = [(mi, s, l, structure.velocity_for(vel_energy, v)) for mi, s, l, v in notes]
            _emit(part, notes, abs_bar * TICKS_PER_BAR)


def _sub_economy(notes, m: Material, prof, root, nxt_root, rel_bar):
    """The 140 drop's sub: a handful of notes, said with weight.

    A dubstep sub moves - it is the riff as often as not - but among very
    few pitches: the chord's root and its octave, the fifth, and in a mode
    with a flat second the b2 the sub leans on and falls back from. One
    note per eight-bar phrase may step out of that, the approach into the
    next chord at the end of the phrase. A walking sub - the riff in the
    bass - keeps the chord's third and seventh as well. Everything else the generators
    reached for - sevenths, passing tones, the tune doubled in the bass - is
    brought onto the nearest of those, keeping its rhythm: the bar keeps its
    groove and loses its wandering. (A roots bassline is a melody, and a
    garage bass bounces between chord tones: this is only the drop sub of
    the dubstep family.)"""
    if not notes:
        return notes
    r = _sub_note(m, prof, root, root)
    allowed = {r % 12, _sub_note(m, prof, root + 4, root) % 12}
    third = None
    if getattr(m.bass, "style", "walk") == "walk":
        # A walking sub is the riff: it keeps the chord's third and its
        # seventh - the b7 below the root is the sub's oldest move - so it
        # walks the chord's own four notes rather than the scale. A stray
        # note equally near the third and the root or fifth goes to the
        # third. Held, pulsing and skipping subs sit on the root by design.
        third = _sub_note(m, prof, root + 2, root) % 12
        allowed |= {third, _sub_note(m, prof, root + 6, root) % 12}
    colour = hook_mod.MODE_COLOUR.get(m.scale.name)
    if colour is not None and colour[0] == 1:        # phrygian's b2 - the wobble's other note
        allowed.add(m.scale.note(1) % 12)
    approach = rel_bar % 8 == 7
    out = []
    for mi, st, ln, v in notes:
        if mi % 12 in allowed or (approach and st >= 14.0):
            out.append((mi, st, ln, v))
            continue
        # Nearest allowed pitch in the sub window; on a tie, a walking
        # line's third, then the root.
        choices = [c for c in range(prof.sub.low, prof.sub.high + 1) if c % 12 in allowed]
        best = min(choices, key=lambda c: (abs(c - mi), c % 12 != third, c % 12 != r % 12))
        out.append((best, st, ln, v))
    return out


def _bass_style(notes, m: Material, prof, root, nxt_root, rel_bar, rng):
    """Turn the generated line into the song's kind of bassline. `walk` is
    the line as generated; the others keep its root and pitch world but
    change what a bar *is*."""
    style = getattr(m.bass, "style", "walk")
    if style == "walk" or not notes:
        return notes
    r = _sub_note(m, prof, root, root)
    fifth = _sub_note(m, prof, root + 4, root)
    vel = max(v for _, _, _, v in notes)
    # Phrases have ends. Bar 4 of every eight bends the figure; bar 8 turns
    # it around and walks into the next phrase - the bassist's job, and the
    # difference between a loop and a line.
    pos = rel_bar % 8
    if pos == 7:
        out = _bass_turnaround(style, m, prof, root, nxt_root, vel)
        if out:
            return out
    seventh = _sub_note(m, prof, root + 6, root)
    if pos == 3:
        if style == "sustain":
            up = r + 12 if r + 12 <= prof.sub.high else fifth
            return [(r, 0.0, 7.5, vel), (up, 8.0, 6.0, vel - 8)]
        if style == "pulse":
            return [(r, 0.0, 3.2, vel), (r, 4.0, 3.2, vel - 8), (fifth, 8.0, 3.2, vel - 8),
                    (seventh, 12.0, 3.2, vel - 8)]
        if style == "skip":
            return [(r, 0.0, 4.5, vel), (fifth, 6.0, 1.5, vel - 14), (r, 10.0, 3.0, vel - 4)]
    if style == "sustain":
        # One held note under the bar; every other bar a pickup on the last
        # eighth leans into the next root.
        out = [(r, 0.0, 14.0 if rel_bar % 2 else 15.5, vel)]
        if rel_bar % 2:
            out.append((_sub_note(m, prof, nxt_root - 1, nxt_root), 14.0, 2.0, vel - 10))
        return out
    if style == "pulse":
        return [(r if k % 2 == 0 else (fifth if k == 3 else r), float(4 * k), 3.2, vel - (0 if k == 0 else 8))
                for k in range(4)]
    if style == "skip":
        # The 2-step bassline: the root with the kick on the one, an octave
        # pop on the & of 2, the root again with the late kick on the & of 3,
        # and every other bar a pickup that leans into the next chord.
        hi = r + 12 if r + 12 <= prof.sub.high else fifth
        out = [(r, 0.0, 4.5, vel), (hi, 6.0, 1.5, vel - 14), (r, 10.0, 3.0, vel - 4)]
        if rel_bar % 2:
            out.append((_sub_note(m, prof, nxt_root - 1, nxt_root), 14.0, 1.5, vel - 12))
        return out
    if style == "bounce":
        hi = r + 12 if r + 12 <= prof.sub.high else r
        return [((r if k % 2 == 0 else hi), float(2 * k), 1.6, vel - (0 if k % 2 == 0 else 14))
                for k in range(8) if not (k == 7 and rel_bar % 2)]
    # sparse: the notes on the downbeat and the half-bar, held.
    keep = [(mi, st, ln, v) for mi, st, ln, v in notes if st in (0.0, 8.0)]
    if not keep:
        keep = [(r, 0.0, 6.0, vel)]
    return [(mi, st, 7.0, v) for mi, st, _, v in keep]


def _bass_turnaround(style, m: Material, prof, root, nxt_root, vel):
    """The last bar of an eight-bar phrase: the figure's first half, then a
    walk - fifth, seventh, a step into the next root - so the bass carries
    the phrase over the barline instead of starting the loop again."""
    r = _sub_note(m, prof, root, root)
    fifth = _sub_note(m, prof, root + 4, root)
    seventh = _sub_note(m, prof, root + 6, root)
    into = _sub_note(m, prof, nxt_root - 1, nxt_root)
    if style == "sustain":
        return [(r, 0.0, 7.5, vel), (fifth, 8.0, 3.5, vel - 8), (seventh, 12.0, 1.8, vel - 10),
                (into, 14.0, 1.8, vel - 6)]
    if style == "pulse":
        return [(r, 0.0, 3.2, vel), (r, 4.0, 3.2, vel - 8), (fifth, 8.0, 1.8, vel - 8),
                (seventh, 10.0, 1.8, vel - 8), (fifth, 12.0, 1.8, vel - 8), (into, 14.0, 1.8, vel - 6)]
    if style == "bounce":
        hi = r + 12 if r + 12 <= prof.sub.high else r
        return [(r, 0.0, 1.6, vel), (hi, 2.0, 1.6, vel - 14), (r, 4.0, 1.6, vel), (hi, 6.0, 1.6, vel - 14),
                (seventh, 8.0, 1.6, vel - 8), (fifth, 10.0, 1.6, vel - 8), (seventh, 12.0, 1.6, vel - 8),
                (into, 14.0, 1.8, vel - 4)]
    if style == "skip":
        return [(r, 0.0, 4.5, vel), (fifth, 6.0, 1.5, vel - 14), (seventh, 10.0, 1.5, vel - 6),
                (fifth, 12.0, 1.5, vel - 8), (into, 14.0, 1.5, vel - 6)]
    if style == "sparse":
        return [(r, 0.0, 7.0, vel), (fifth, 8.0, 5.0, vel - 8), (into, 14.0, 1.8, vel - 6)]
    return None


def _lead_busy_beats(m: Material, section: Section) -> set[int]:
    """Beats (section-relative, 4 steps each) where the lead, as performed -
    graces, mordents and all - plays two or more notes. That beat is the
    lead's; everyone else holds."""
    lead = m.built.get("lead")
    if lead is None or not section.plays("lead"):
        return set()
    count: dict[int, int] = {}
    for n in lead.notes:
        if section.start_tick <= n.start < section.start_tick + section.length_ticks:
            b = (n.start - section.start_tick) // (4 * TICKS_PER_16TH)
            count[b] = count.get(b, 0) + 1
    return {b for b, c in count.items() if c >= 2}


def _yield_to_lead(notes, busy: set[int], lo_step: float, space: float = 1.0,
                   gate: float = 0.9):
    """The bass gives the lead's figures room: inside a beat the lead is
    busy, only a note on the beat itself survives (and holds through it);
    the bass moves again when the lead holds or rests."""
    if not busy:
        return notes
    out = []
    for mi, st, ln, v in notes:
        beat = int((lo_step + st) // 4)
        # `space` is how readily the band gives way - per beat, reproducibly.
        yields = beat in busy and ((beat * 7919) % 100) / 100.0 < space
        if yields and (lo_step + st) % 4 != 0:
            continue
        out.append((mi, st, ln, v))
    # Re-join: each kept note holds to the next one (mono; capped at a bar).
    out.sort(key=lambda n: n[1])
    joined = []
    for i, (mi, st, ln, v) in enumerate(out):
        nxt = out[i + 1][1] if i + 1 < len(out) else st + ln
        if int((lo_step + st) // 4) in busy:
            ln = max(ln, min(nxt - st, 8.0) * gate)
        joined.append((mi, st, ln, v))
    return joined


def _on_eighths(notes, keep: frozenset[int] = frozenset()):
    """The sub plays on the eighth-note grid. A sub bass on the in-between
    sixteenths is heard as out of
    time, not as syncopation: the low end has to be where the kick is. (Not
    on the sixteenths in `keep`: in roots and steppers, the ones their records
    play about as often as the off-eighths - Profile.sub_sixteenths.) A note
    on an odd sixteenth moves forward to the next eighth (or back, but never
    onto the downbeat, which belongs to the root); if both are taken, it
    goes."""
    taken = {float(int(round(st))) for _, st, _, _ in notes if int(round(st)) % 2 == 0}
    out, used = [], set()
    for mi, st, ln, v in sorted(notes, key=lambda n: n[1]):
        s = int(round(st))
        if s % 2 == 0 or s in keep:
            q = float(s)
        elif s + 1 < STEPS_PER_BAR and float(s + 1) not in taken:
            q = float(s + 1)               # forward to the next eighth
        elif s - 1 > 0 and float(s - 1) not in taken:
            q = float(s - 1)               # back - but never onto the downbeat
        else:
            continue
        if q in used:
            continue
        used.add(q)
        out.append((mi, q, max(0.5, ln), v))
    return out


def _sub_note(m: Material, prof, degree: int, root: int) -> int:
    """A bass degree placed relative to the root's home in the sub window, so
    an octave jump stays a jump instead of folding back onto the root."""
    home = pitch.transpose_into(m.scale.note(root), prof.sub.low, prof.sub.low + 11)
    note = home + (m.scale.note(degree) - m.scale.note(root))
    while note > prof.sub.high:
        note -= 12
    while note < prof.sub.low:
        note += 12
    return note


def _bass_from_hook(tones, lo_step, root, m: Material, prof, section: Section):
    """The hook's rhythm, spoken by the sub - the riff in unison.

    Strong beats say the root: the bass still owns the harmony. Everywhere
    else the sub doubles the hook's own note two or three octaves down, so
    the bass and the tune move as one figure; the hook's high point pops
    the root's octave instead, because a sub cannot follow a tune that far
    up. The downbeat is always there - kick and sub land together - even
    when the hook enters late. Note lengths follow the bass's gate.
    """
    ordered = sorted(tones, key=lambda t: t.step)
    top = max(t.degree for t in ordered)
    out = []
    if ordered[0].step - lo_step >= 2:
        out.append((0.0, None, root))
    for t in ordered:
        step = t.step - lo_step
        if metric_weight(int(step) % 16) >= 3:
            out.append((step, None, root))
        elif t.degree == top:
            out.append((step, None, root + 7))
        else:
            out.append((step, t, None))
    notes = []
    home = pitch.transpose_into(m.scale.note(root), prof.sub.low, prof.sub.low + 11)
    for i, (step, tone, degree) in enumerate(out):
        nxt = out[i + 1][0] if i + 1 < len(out) else 16.0
        length = max(0.75, (nxt - step) * m.bass.gate)
        vel = 112 if metric_weight(int(step) % 16) >= 3 else 100
        if tone is not None:
            note = _borrowed(tone, m, tone.step, section)
            while note - home > 9:
                note -= 12
            while note < home - 3:
                note += 12
            # Fold, never clamp: clamping moves a note to a different pitch.
            note = pitch.transpose_into(note, prof.sub.low, prof.sub.high)
        else:
            note = _sub_note(m, prof, degree, root)
        notes.append((note, step, length, vel))
    return notes


# How the sub says a melody's moves: the figure's steps become the bass's
# own vocabulary - root, the b7 below, the fifth, the octave.
def _bass_degree(rel: int) -> int:
    if rel <= -2:
        return -3          # the fifth below
    if rel == -1:
        return -1          # the b7 below, leading home
    if rel == 0:
        return 0
    if rel == 1:
        return 2           # a step up is the third: colour, not just the frame
    if rel <= 3:
        return 4           # the fifth
    return 7               # the octave


def _sub_reply(notes, replies, lo_step, root, m: Material, prof):
    """The bass answers the call: in the space after it, the sub plays the
    call's figure back - same rhythm, the shape in bass moves - and its own
    pattern stands aside so the reply is heard."""
    out = list(notes)
    for a, b, _, fig in replies:
        wa, wb = a - lo_step, b - lo_step
        kept = []
        for mi, st, ln, v in out:
            if wa <= st < wb and not (st == 0 and wa < 0):
                # The reply owns the window - but a reply carried over the
                # barline never takes the new bar's downbeat: that is where
                # the bass states the chord.
                continue
            if st < wa < st + ln:
                ln = max(0.5, wa - st)
            kept.append((mi, st, ln, v))
        out = kept
        onsets = [(wa + rel, deg) for rel, deg in fig if wa + rel < min(wb, STEPS_PER_BAR)
                  and wa + rel >= 0]
        for k, (st, deg) in enumerate(onsets):
            nxt = onsets[k + 1][0] if k + 1 < len(onsets) else min(wb, STEPS_PER_BAR)
            ln = max(0.75, (nxt - st) * max(m.bass.gate, 0.92))
            note = _sub_note(m, prof, root + _bass_degree(deg), root)
            out.append((note, float(st), ln, 112 if k == 0 else 100))
    return sorted(out, key=lambda n: n[1])


def _answer_breaths(notes, line: Line, lo_step, nxt_root, m: Material, prof, rng):
    """Where the lead breathes and the sub is silent, a pickup into the next
    downbeat: the step below the coming root, then the root's neighbour -
    the bass leaning into the next call."""
    if not line.answers:
        return notes
    out = list(notes)
    for a, b in line.answers_in(lo_step, lo_step + STEPS_PER_BAR):
        a, b = a - lo_step, b - lo_step
        if b - a < 3 or b < 15 or rng.random() > m.interplay.answer * 0.6:
            continue
        if any(a <= s < b for _, s, _, _ in out):
            continue
        pick = rng.choice((-1, 1, 4))
        out.append((_sub_note(m, prof, nxt_root + pick, nxt_root), b - 2.0, 1.6, 96))
    return sorted(out, key=lambda n: n[1])


# The most stabs in a chord answer to the lead (see _build_chords).
ANSWER_STABS = 3


def _build_chords(part, section, prof, m: Material, rng):
    """Offbeat stabs that leave the lead room - and answer it.

    `space` ducks a stab that would land on one of the lead's notes: the
    stab and the melody are not allowed to say the same thing at the same
    time. `answer` fills the lead's breaths with the hook's own rhythm on
    the chord: the call, then the band's reply.
    """
    line = m.line(section)
    lead_on = section.plays("lead")
    onsets = line.onsets() if lead_on else set()
    busy = _lead_busy_beats(m, section)
    ip = m.interplay
    for bar in range(section.bars):
        bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
        ev = m.voiced.at(bar_tick)
        if ev is None:
            continue
        energy = structure.bar_energy(section, bar)
        density = structure.density_for(energy, "chords")
        p = rhythm.thin(m.stab, density, rng) if density < 1.0 else m.stab
        lo = bar * STEPS_PER_BAR
        if onsets:
            p = rhythm.mask(p, lambda st: not (
                (lo + int(round(st))) in onsets and rng.random() < ip.space))
        hits = [(h.step, h.length, h.velocity) for h in p.sorted_hits()]
        if busy:
            # Under a lead figure the skank says its one offbeat and no more.
            kept, seen = [], set()
            for h in hits:
                beat = int((lo + h[0]) // 4)
                if beat in busy:
                    if beat in seen:
                        continue
                    seen.add(beat)
                    h = (h[0], h[1], h[2] * 0.85)
                kept.append(h)
            hits = kept
        # Phrase ends. Bar 4 of eight pushes into the next bar: a stab on
        # the & of 4 that already plays the coming chord (the anticipation).
        # Bar 8 leaves its last beat empty for the drums' turnaround. And in
        # the second half of a long drop the voicing lifts an inversion - the
        # same chords, brighter, which a listener hears as the drop growing.
        anticipate = False
        if section.bars >= 8:
            pos = bar % 8
            if pos == 3 and not any(h[0] >= 13 for h in hits) \
                    and (lo + 14) not in onsets and (lo + 14) // 4 not in busy:
                hits.append((14.0, max(prof.skank_length, 0.5), 96))
                anticipate = True
        lift_all = structure.kind_of(section) == "peak" and section.bars >= 16 \
            and bar >= section.bars // 2
        lifts = set()
        if lead_on:
            # The chords' turn to answer: the call's figure, stabbed on the
            # chord, the voicing lifted where the figure rises - and the
            # ordinary stabs stand aside inside the window.
            for a, b, _, fig in line.response_in(lo, lo + STEPS_PER_BAR, "chords"):
                wa, wb = a - lo, b - lo
                # The answer is the call's figure as a keys player answers
                # it: its strong points only - on the eighths, an eighth
                # apart at least, three stabs at most - and said once. A run
                # of sixteenth chords, said again as a "call-back", read as an
                # echo through the preview's delay; on a real synth with its
                # own release, swung by the Deluge, it was a pile-up. The
                # echo is the delay's job, not the notes'. Then a beat of air,
                # and the skank comes back in.
                stabs: list[tuple[int, int]] = []
                for rel, deg in fig:
                    st = wa + rel
                    st += st % 2                     # onto the eighth grid
                    if st >= min(wb, STEPS_PER_BAR) or st < 0:
                        continue
                    if stabs and st - stabs[-1][0] < 2:
                        continue
                    stabs.append((st, deg))
                    if len(stabs) == ANSWER_STABS:
                        break
                if not stabs:
                    continue
                # a beat of air after the answer before the skank returns
                own = stabs[-1][0] + 4
                # ...and an eighth clear before it, so it stands apart from
                # the skank (a double chop's pair would run into it)
                clear = stabs[0][0] - 2
                hits = [h for h in hits if not clear <= h[0] < own]
                for st, deg in stabs:
                    hits.append((st, max(prof.skank_length * 1.6, 0.6), 100))
                    if deg > 0:
                        lifts.add(st)
        if section.bars >= 8 and bar % 8 == 7:
            hits = [h for h in hits if h[0] < 12]
        if not hits:
            continue
        ordered = sorted({h[0]: h for h in hits}.values())
        pat = rhythm.Pattern(STEPS_PER_BAR, [rhythm.Hit(s, int(v), l) for s, l, v in ordered])
        # Velocity jitter seeded by position, not by the shared stream: a new
        # lead must not re-roll the dynamics of every stab after it.
        pat = rhythm.humanize(pat, 0.0, 7, random.Random(f"{m.seeds.get('chords')}:{section.start_bar + bar}"))
        chord = m.voiced.at(bar_tick)
        for hit in pat.sorted_hits():
            ev_here = m.voiced.at(bar_tick + int(hit.step * TICKS_PER_16TH)) or chord
            if anticipate and hit.step >= 14:
                ev_here = m.voiced.at(bar_tick + TICKS_PER_BAR) or ev_here
            pitches = sorted(ev_here.chord.pitches)
            if lift_all and pitches[0] + 12 <= prof.chords.high:
                pitches = pitches[1:] + [pitches[0] + 12]     # the lift
            if int(round(hit.step)) in lifts and pitches[0] + 12 <= prof.chords.high:
                pitches = pitches[1:] + [pitches[0] + 12]     # next inversion up
            for note in pitches:
                _emit(part, [(note, hit.step, hit.length,
                              structure.velocity_for(energy, hit.velocity))], bar_tick)


def _build_bubble(part, section, prof, m: Material):
    """The organ bubble: the reggae keyboard's shuffle under the skank.

    Each beat, a quiet sixteenth then the loud '&' (rhythm.bubble), on the
    chord's upper voices in the middle of the keys - the root is the bass's.
    At low energy only the '&'s; the last beat of an eight-bar phrase is left
    to the drums' turnaround, as the skank leaves it. Velocities wander by
    position, never by the shared stream, so the same bar plays the same.
    """
    pat = rhythm.bubble(STEPS_PER_BAR, 84)
    for bar in range(section.bars):
        bar_tick = (section.start_bar + bar) * TICKS_PER_BAR
        energy = structure.bar_energy(section, bar)
        jitter = random.Random(f"{m.seeds.get('pad')}:{section.start_bar + bar}")
        for hit in pat.sorted_hits():
            soft = hit.velocity < 70
            if soft and energy < 0.45:
                continue
            if section.bars >= 8 and bar % 8 == 7 and hit.step >= 12:
                continue
            ev = m.voiced.at(bar_tick + int(hit.step * TICKS_PER_16TH))
            if ev is None:
                continue
            voices = sorted({pitch.transpose_into(n, prof.pad.low, prof.pad.high)
                             for n in ev.chord.pitches})
            root_pc = ev.chord.root_pc
            upper = [v for v in voices if v % 12 != root_pc] or voices
            vel = structure.velocity_for(energy, hit.velocity + jitter.randint(-6, 6))
            for note in upper:
                _emit(part, [(note, hit.step, hit.length, vel)], bar_tick)


def _build_pad(part, section, prof, m: Material, rng):
    """Held chords, with a top voice that moves against the tune.

    The counter-line is a guide-tone line: on each chord, the chord tone (or
    its seventh or ninth) nearest the last one, preferring to move against
    the direction the lead is moving in. Two lines in contrary motion is the
    oldest trick in counterpoint and the cheapest way to make a pad sound
    composed rather than held down.
    """
    if getattr(prof, "pad_style", "held") == "bubble":
        _build_bubble(part, section, prof, m)
        return
    start, end = section.start_tick, section.start_tick + section.length_ticks
    line = m.line(section)
    lead = sorted(line.tones, key=lambda t: t.step) if section.plays("lead") else []
    prev_top: int | None = None
    prev_lead: float | None = None
    for ev in m.voiced.events:
        if ev.end <= start or ev.start >= end:
            continue
        s, e = max(ev.start, start), min(ev.end, end)
        if e - s <= 0:
            continue
        vel = structure.velocity_for(section.energy, 66)
        voices = sorted({pitch.transpose_into(n, prof.pad.low, prof.pad.high)
                         for n in ev.chord.pitches})
        hold = max(1, e - s - TICKS_PER_16TH)
        counter = rng.random() < m.interplay.counter

        # Which way is the tune going over this chord?
        lo_step = (s - start) / TICKS_PER_16TH
        hi_step = (e - start) / TICKS_PER_16TH
        here = [t.midi(m.scale) for t in lead if lo_step <= t.step < hi_step]
        lead_mean = sum(here) / len(here) if here else prev_lead
        direction = 0
        if lead_mean is not None and prev_lead is not None:
            direction = 1 if lead_mean > prev_lead + 0.5 else (-1 if lead_mean < prev_lead - 0.5 else 0)
        if here:
            prev_lead = lead_mean

        # The counter-line *is* the chord's top voice: the lower voices hold,
        # the top one moves - inside the pad's window, never doubling a held
        # voice, and never a semitone from what the lead holds over it.
        base = voices[:-1] if len(voices) >= 3 else voices
        held = {t.midi(m.scale) % 12 for t in lead if lo_step <= t.step < hi_step}
        lo_top = max(base) + 1
        pcs = {m.scale.note(ev.chord.degree + d) % 12 for d in (0, 2, 4, 6)}
        cands = [c for c in range(lo_top, prof.pad.high + 1)
                 if c % 12 in pcs and c not in base]
        cands = [c for c in cands if not any((c - h) % 12 in (1, 11) for h in held)] or cands
        if section.plays("chords"):
            # The stabs already say the chord. The pad does not say it again
            # in the same register: it keeps one voice - its moving line -
            # and leaves the rest of the space to the stabs.
            if not cands:
                part.add(Note(voices[-1], s, hold, vel - 6))
                continue
            base = []
        elif not counter or not cands:
            for note in voices:
                part.add(Note(note, s, hold, vel))
            continue
        for note in base:
            part.add(Note(note, s, hold, vel))
        if prev_top is None:
            top = min(cands, key=lambda c: abs(c - (lo_top + 4)))
        else:
            def cost(c, pt=prev_top):
                move = c - pt
                against = 0 if direction == 0 else (2 if move * direction > 0 else 0)
                return abs(move) + against + (3 if move == 0 else 0) * 0.5
            top = min(cands, key=cost)
        # The suspension: across a chord change the old top note is held
        # into the new chord - a dissonance - and resolves down a step a beat
        # or two later (the 4-3, 9-8, 7-6 of every chorale). Only where the
        # held note is not already in the new chord and does not rub against
        # the tune. The counter dial sets how often.
        s_top = s
        if prev_top is not None and e - s >= TICKS_PER_BAR \
                and rng.random() < 0.25 + 0.6 * m.interplay.counter:
            for c in sorted(cands, key=lambda c: abs(c - top)):
                if 1 <= prev_top - c <= 2 and prev_top % 12 not in pcs \
                        and not any((prev_top - h) % 12 in (1, 11) for h in held):
                    hang = 4 * TICKS_PER_16TH * (1 if e - s < 2 * TICKS_PER_BAR else 2)
                    part.add(Note(prev_top, s, hang, vel + 10))
                    top, s_top = c, s + hang
                    break
        span = e - s_top
        if span >= 2 * TICKS_PER_BAR and m.interplay.counter > 0.6:
            # A long chord gets a second guide tone halfway: a step to the
            # nearest other candidate, so the line keeps moving.
            half = s + ((e - s) // 2 // TICKS_PER_BAR) * TICKS_PER_BAR
            second = min((c for c in cands if c != top), key=lambda c: abs(c - top), default=top)
            part.add(Note(top, s_top, max(1, half - s_top - TICKS_PER_16TH), vel + 8))
            part.add(Note(second, half, max(1, e - half - TICKS_PER_16TH), vel + 8))
            prev_top = second
        else:
            part.add(Note(top, s_top, max(1, e - s_top - TICKS_PER_16TH), vel + 8))
            prev_top = top


def _build_lead(part, section, prof, m: Material, rng):
    """The theme plan for this section, sung in the lead's register and
    *performed*.

    The plan (arrange.interplay) decides what the hook does here - tease,
    climbing fragment, full statement, augmented memory, dissolve. This
    places it (one octave shift for the whole song, so the hook keeps its
    shape; velocity from the bar's energy, so a ramped build arrives with
    its melody) and then plays it, through the performance dials:

    gate      how much of each gap a note holds - staccato riff to legato song
    slide     step moves played overlapping, so a mono synth glides
    accent    the riff's strong notes hit, its weak ones ghost
    grace     a 32nd flicked in from a step below into strong notes
    flick     weak notes popped up an octave
    echo      the last note repeated at a dotted eighth into a breath, fading
    ornament  a trill on long notes
    """
    lp = m.lead
    throw = 1.5 if section.dub_throw else 1.0
    ordered = sorted(m.line(section).tones, key=lambda t: t.step)
    notes = []   # (midi, step, length, velocity)
    for t in ordered:
        midi = _borrowed(t, m, t.step, section) + m.lead_shift
        lo, hi = lead_window(prof, lp)
        while midi > hi:
            midi -= 12
        while midi < lo:
            midi += 12
        notes.append([midi, t.step, t.length, t.velocity])

    out = []      # [midi, step, length, velocity, slides]
    base_seed = m.lead_seed * 1000003
    # The hook's cycle: a performance decision belongs to a place *in the
    # hook*, so every pass of it - in this drop and the next - is played the
    # same way. A riff is written once, flick and grace and all, and then
    # it repeats; a player who reshuffles the ornaments on every pass is a
    # random number generator, and that is exactly what a listener hears.
    cycle = m.hook.length if m.hook is not None and m.hook.length > 0 else 4 * STEPS_PER_BAR
    for i, (midi, step, length, vel) in enumerate(notes):
        # Each note's performance is decided by where it is, not by how many
        # notes came before it: change the hook and the notes that stay keep
        # their accents, flicks and graces.
        # (Position only, never pitch: a transposed hook is played the same.)
        rng = random.Random(base_seed + int(round((step % cycle) * 4)) * 31)
        nxt = notes[i + 1] if i + 1 < len(notes) else None
        gap = (nxt[1] - step) if nxt is not None else length
        bar = int(step // STEPS_PER_BAR)
        energy = structure.bar_energy(section, min(bar, section.bars - 1))
        w = metric_weight(int(step) % 16) / 4.0
        # Accent: contrast around the middle, not loudness - the strong notes
        # rise, the weak ones ghost, and nothing is pushed into the ceiling.
        shaped = 80 + (vel - 80) * (1 - 0.5 * lp.accent) + lp.accent * (w - 0.35) * 55
        v = structure.velocity_for(energy, shaped) + rng.randint(-3, 3)

        # Gate: a sixteenth-note riff at gate 0.3 is all bite; at 1.0 it sings.
        held = max(0.4, min(length, gap) * (0.3 + 0.7 * lp.gate))
        # Phrase-final lengthening: the note a figure lands on - the one a
        # breath follows - is held out, however clipped the notes before it.
        # A player shortens the run and sings the landing; a riff that chops
        # its last note too stops instead of arriving. (Stabs excepted: the
        # hit is the point.)
        if gap >= 4.0 and length >= 1.0 and m.hook is not None \
                and m.hook.params.phrasing != "stabs":
            held = max(held, min(length, gap) * 0.85)
        slides = False
        # (A glide needs the notes joined: staccato and slide do not mix.)
        if nxt is not None and gap <= 4.0 and 0 < abs(nxt[0] - midi) <= 5 \
                and rng.random() < lp.slide * min(1.0, lp.gate * 1.6):
            held, slides = gap + 0.25, True              # overlap: the synth glides

        # Flick: a weak note jumps the octave - only where it fits.
        # (Not in a build: a build approaches the drop's high point, it
        # does not leap past it.)
        # And only to a height the line can reach: a flicked note stays
        # within an octave of the notes either side of it, so the gesture is
        # the octave pop a player makes, not a leap off the top of the tune.
        prev_m = notes[i - 1][0] if i > 0 else midi
        next_m = nxt[0] if nxt is not None else midi
        if w < 0.5 and rng.random() < lp.flick and midi + 12 <= lead_window(prof, lp)[1] + 5 \
                and structure.kind_of(section) != "build" \
                and abs(midi + 12 - prev_m) <= 12 and abs(midi + 12 - next_m) <= 12:
            midi += 12

        # Grace: a 32nd from a scale step below into a strong note, stealing
        # the time from the note before if it has any to spare.
        # Grace notes go where a player puts them: into a leap, or into a
        # note that starts a figure after a rest - not on every note.
        leap_in = i > 0 and abs(midi - notes[i - 1][0]) >= 3
        after_rest = i == 0 or step - (notes[i - 1][1] + notes[i - 1][2]) >= 1.5
        if (leap_in or after_rest) and rng.random() < lp.grace:
            prev_end = out[-1][1] + out[-1][2] if out else -1.0
            if prev_end > step - 0.5 and out and out[-1][1] <= step - 1.0:
                out[-1][2] = step - 0.5 - out[-1][1]
                out[-1][4] = False
                prev_end = step - 0.5
            if prev_end <= step - 0.5 and step >= 0.5:   # never before the section
                below = m.scale.snap(midi - 1, prefer="down")
                out.append([below, step - 0.5, 0.45, max(30, int(v * 0.7)), False])

        # Ornament: a trill on a long note; on a short one, a mordent - note,
        # upper neighbour, note, in 32nds.
        space = min(length, gap)     # what the note has to play with, gate aside
        if rng.random() < lp.ornament and space >= 2.0:     # landing notes, eighths
            upper = m.scale.snap(midi + 1, prefer="up")
            flicks = (midi, upper, midi, upper) if space >= 3.0 else (midi, upper)
            for k, pitch_ in enumerate(flicks):
                out.append([pitch_, step + k * 0.5, 0.45, max(30, v - (0 if k == 0 else 12)), False])
            rest = len(flicks) * 0.5
            out.append([midi, step + rest, max(0.4, held - rest), max(30, v - 6), slides])
        else:
            out.append([midi, step, held, v, slides])

        # Echo: the note again at a dotted eighth, twice, fading - printed
        # into the breath that follows, as the desk would throw it.
        room = (nxt[1] if nxt is not None else section.bars * STEPS_PER_BAR) - step
        # The echo gives way to a chord reply (same register); a bass reply
        # is two octaves down and the echo can ring over it.
        answered = bool(m.line(section).response_in(step + 2, step + 7, "chords"))
        if room >= 6 and not answered and rng.random() < min(1.0, lp.echo * throw):
            for k, scale_v in ((1, 0.55), (2, 0.3)):
                if 3 * k + 1 <= room:
                    out.append([midi, step + 3 * k, min(held, 2.0), max(20, int(v * scale_v)), False])

    # Swing the starts, then make sure only an intended slide overlaps.
    out.sort(key=lambda n: n[1])
    lo, hi = lead_window(prof, lp)
    for n in out:   # a grace below the floor, a trill above a flick: fold back in
        while n[0] > hi + 5:
            n[0] -= 12
        while n[0] < lo:
            n[0] += 12
    starts = [n[1] for n in out]
    for i, (midi, step, length, v, slides) in enumerate(out):
        start = starts[i]
        end = start + max(0.4, length)
        if i + 1 < len(out) and not slides:
            end = min(end, starts[i + 1])
        _emit(part, [(midi, start, max(0.3, end - start), v)], section.start_tick)


def _lead_line(part: Part) -> None:
    """One voice: same-start notes keep the louder, overlaps are cut back to
    the glide - a sixteenth's quarter - that a slide asked for."""
    notes = sorted(part.notes, key=lambda n: (n.start, -n.velocity))
    out: list[Note] = []
    for n in notes:
        if out and n.start == out[-1].start:
            continue
        if out and out[-1].end > n.start + TICKS_PER_16TH // 4:
            prev = out[-1]
            out[-1] = Note(prev.midi, prev.start, max(1, n.start - prev.start), prev.velocity)
        out.append(n)
    part.notes = out


def _button(song: Song, arrangement: Arrangement, m: Material, prof) -> None:
    """End on a hit, not a stop: kick, crash and the tonic in the sub on the
    downbeat after the last bar. It sits past the arrangement, so a looping
    Deluge clip never plays it; a MIDI file or a WAV ends on it."""
    last = arrangement.sections[-1] if arrangement.sections else None
    if last is None:
        return
    end = arrangement.length_ticks
    k = m.lift[1] if m.lift and m.lift[1] else 0     # a lifted song ends in its new key
    for part in song.parts:
        if part.role == "drums" and last.plays("drums"):
            song.coda.append(("drums", Note(KIT_ROWS["kick"], end, 22, 118)))
            song.coda.append(("drums", Note(KIT_ROWS["crash"], end, TICKS_PER_BAR, 104)))
        elif part.role == "sub" and last.plays("sub"):
            tonic = pitch.transpose_into(m.scale.root + k, prof.sub.low, prof.sub.low + 11)
            song.coda.append(("sub", Note(tonic, end, TICKS_PER_BAR, 112)))
        elif part.role == "pad" and last.plays("pad"):
            ev = m.voiced.events[0] if m.voiced.events else None
            if ev is not None and ev.chord.degree == 0:
                for n in ev.chord.pitches:
                    song.coda.append(("pad", Note(
                        pitch.transpose_into(n + k, prof.pad.low, prof.pad.high),
                        end, TICKS_PER_BAR * 2, 70)))


def _borrowed(t, m: Material, step: float, section: Section | None) -> int:
    """A tone's pitch, keeping a chromatic alteration only while the chord
    that lent it is still sounding. A borrowed leading tone held into the
    next chord - or ornamented, or lifted an octave into a bar where the
    harmony has moved on - goes back to its scale degree."""
    midi = t.midi(m.scale)
    if not t.alter:
        return midi
    base = section.start_tick if section is not None else 0
    ev = m.plan.at(base + int(round(step * TICKS_PER_16TH)))
    if ev is not None and ev.chord.contains_pc(midi):
        return midi
    return midi - t.alter


def _apply_silences(part: Part, silences) -> None:
    """Cut every note in a silence window; truncate notes ringing into one.
    The riser is the exception: it is what the silence is made of."""
    if not silences:
        return
    kept = []
    for n in part.notes:
        if part.output == "kit" and n.midi == KIT_ROWS["riser"]:
            kept.append(n)
            continue
        dead = False
        for a, b, keep in silences:
            if part.role in keep:
                continue
            if a <= n.start < b:
                dead = True
                break
            if n.start < a < n.end:
                n = Note(n.midi, n.start, max(1, a - n.start), n.velocity)
        if not dead:
            kept.append(n)
    part.notes = kept


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


# The kit's desk moves (Deluge kit clips, firmware ai-kit-automation): the
# same moves the audio preview's desk makes. Values are 0..1 of the Deluge
# knob. A Deluge sound's delay knob IS its feedback, linearly (knob 50 =
# 1.0, repeating for ever) - and a jam loops a build or a throw clip, so a
# high value stacks up into a wall that rings on after the stop. 0.4 keeps
# about seven audible repeats (Delay::setTimeToAbandon); nothing goes past
# DELAY_CEILING. The kit presets set the backbeat's send at 0.14 in dub.
DELAY_CEILING = 0.4
KIT_SWEEP = (0.0, 0.6)          # whole-kit HPF across a build's last two bars
SNARE_THROW = 0.3               # the backbeat's echo opened in a dub throw
SNARE_RUNAWAY = (0.18, 0.4)     # ... and running on into a drop
SNARE_THROW_REVERB = 0.4
assert max(SNARE_THROW, *SNARE_RUNAWAY) <= DELAY_CEILING


def _add_kit_automation(part: Part, role: str, arrangement: Arrangement, prof: Profile):
    """The desk on the drums: the high-pass sweep that thins the kit through
    the end of every build into a drop, the backbeat's echo opened up in a
    dub throw, and that echo running away over a build's last two bars. Kit
    clips only - per drum where the move is a channel's, whole-kit where it
    is the desk's. (Written to the Deluge by firmware ai-kit-automation;
    older firmware refuses kit automation and the moves are dropped.)"""
    if part.output != "kit":
        return
    sections = arrangement.sections
    two_bars = 2 * TICKS_PER_BAR
    backbeat = KIT_ROWS["snare"]
    for i, section in enumerate(sections):
        if not section.plays(role):
            continue
        following = sections[i + 1] if i + 1 < len(sections) else None
        end = section.start_tick + section.length_ticks
        into_drop = (following is not None and following.energy >= 0.8
                     and section.energy < following.energy)
        if section.dub_throw and role == "drums":
            part.automation.append(Automation(
                "delay", section.start_tick, section.length_ticks, SNARE_THROW, row=backbeat))
            part.automation.append(Automation(
                "reverb", section.start_tick, section.length_ticks, SNARE_THROW_REVERB, row=backbeat))
        if into_drop and section.length_ticks >= two_bars:
            start = end - two_bars
            part.automation.append(Automation("hpf", start, two_bars, *KIT_SWEEP))
            if role == "drums":
                part.automation.append(Automation(
                    "delay", start, two_bars, *SNARE_RUNAWAY, row=backbeat))


def _performance_notes(prof: Profile, m: Material, arrangement: Arrangement, song: Song):
    throws = [s.name for s in arrangement.sections if s.dub_throw]
    midi_parts = sorted((p for p in song.parts if p.output == "midi"), key=lambda p: p.channel)
    automated = any(p.automation for p in song.parts if p.output == "synth")
    kit_moves = sum(len(p.automation) for p in song.parts if p.output == "kit")
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
    ] + ([f"Kits: {', '.join(sorted({p.preset for p in song.parts if p.output == 'kit' and p.preset}))}"
          f" - {kit_moves} desk moves on the drums (sweeps into drops, echo throws)."]
         if kit_moves else [])

# Lead generation — review and improvement plan

What the lead is supposed to be, what it actually does, and the changes to get
it there. The goal, in the maker's words: **the lead needs dials to move it,
it needs proper musical structure, and it needs to move with the rest of the
song.**

---

## 1. How the lead works today

Pipeline (`compose.compose`):

1. `_resolve_material` draws or resolves `LeadParams` (density, syncopation,
   span, octave, step_bias, ornament, contour) and builds **one motif per
   song**: a Euclidean one-bar rhythm (pulse count from `density`, rotation
   from `syncopation`) with degrees placed by `melody.generate` (step bias,
   leap recovery, chord-tone snap on strong beats, contour-following,
   shaped velocity).
2. Each section that plays `lead` gets the motif passed through
   `melody.develop(motif, section.development, rng, scale)` — a fixed 11-step
   ladder (statement → thinned → displaced → transposed → ornamented →
   dropped → fragmented → inverted → raised → augmented → retrograde), where
   stage = the section's index.
3. `_build_lead` then speaks it in one of three styles
   (`call_response` / `long_tones` / `riff`):
   - `call_response`: the phrase is placed every 2–3 bars; odd placements are
     transposed down a step (the "response"); every placement is transposed
     by the current chord's root degree.
   - `riff`: `sequence(phrase, [0, -1])` placed every 4–6 bars.
   - `long_tones`: one or two held tones per bar, degrees taken cyclically
     from the developed motif, first tone snapped to the chord.
   The register is clamped to `profile.lead` (G3–G5) on emission.

**What is genuinely good and should be kept:**

- The motif-plus-development model is the right one: one cell, and the song
  is a schedule of transformations of it. `melody.py` is a proper theory
  module — diatonic space, leap recovery, the development menu.
- Per-part seeding means the lead has its own stream and can be re-rolled
  alone.
- The critic already has a lead check (`_lead_chord_tones`) and a register
  check that catches lead notes escaping the window.
- The working tree (uncommitted) already added most of the lead *dials*:
  `LeadParams`, six `lead.*` knobs in `direct._BOUNDS`, per-section
  `energy`/`ramp`/`bars` steerable paths, velocity shaping in `generate`,
  octave-aware `develop()`, the critic's fix/`revise` machinery, and the UI
  check panel.

**Where it falls short — the three asks, in order:**

### A. Dials — half-built, and part of it is invisible

- **Blocking bug: the lead no longer reproduces.** The working tree draws
  `LeadParams` from `lead_rng` *and then* `melody.generate` uses that same
  stream (`compose._resolve_material`). On a first compose `params.lead` is
  `None` → 7 resolution draws shift the stream; on a recompose it is set →
  the stream starts fresh. Same seed, different motif
  (`[-2,-1,3,2,2,2,-2]` vs `[-2,-1,0,0,1,2,1]` for seed 1). This breaks the
  whole twist model — 3 tests fail
  (`test_resolved_params_reproduce_the_song_exactly`,
  `test_song_diff_reports_which_parts_moved`, `test_state_regenerate_is_local`).
  The bass doesn't have this bug: `params.bass` is drawn from the *shared*
  rng; `bass_rng` is used for the line only.
- `lead_style` (call_response / long_tones / riff) is not steerable at all —
  no knob, no op path, no intent. It lives only in the advanced UI panel
  (a full-recompose select). It is the single biggest character dial.
- `lead.contour` (8 named shapes) is likewise not steerable.
- **The six float knobs that do exist never render.** The UI page's `GROUPS`
  list has no `lead` panel and `PRETTY` has no lead labels, so
  `renderSound` filters them out of every panel. The server serves them in
  `/api/knobs`; the page simply has no place to put them.
- `INTENTS` has no lead direction: "busier"/"thinner" hit drums and bass
  only. A maker saying "lead busier" gets "did not understand".
- The critic's lead finding offers only a **roll**; there is no twist path,
  so the revision loop can never fix a drifting lead without a new take.

### B. Musical structure — one bar, no return, no punctuation

- The motif is exactly one bar (16 steps). Phrases are that cell re-placed;
  even `augment` only makes a 2-bar phrase at one stage. A lead with a
  real phrase (an A/A' two-bar cell) would stop reading as a loop.
- Development stage = section index, a fixed ladder, and it **never comes
  back**. An 8-section song ends on stage 7 (inverted). Songs end on the
  material they began with; the outro restates the statement (possibly
  displaced or octave-dropped), it does not premiere the retrograde.
- Development is not steerable: `SectionSpec` doesn't even carry a
  `development` field — edited sections get stage = their list index. The
  maker cannot say "this section plays the fragmented cell".
- Phrases are truncated at section ends (`room` clamping) but never
  *closed*. Drums got phrase-ending fills in the latest commit; the lead just
  stops mid-phrase. It needs the lead version: a small closing figure (or a
  held chord tone) in the final bar of a section it plays.
- The octave development stages (`dropped`/`raised`) are often silent in
  practice: ±1 octave lands back in the same two-octave window and
  `transpose_into` folds it to the same pitches. The register move is the
  loudest development there is; it should be audible or it shouldn't be on
  the ladder.

### C. Moves with the rest of the song — it doesn't

- **Flat per section, where everything else is per bar.** `_build_lead`
  computes density and velocity from `section.energy` once. Drums use
  `structure.bar_energy(section, bar)` *every bar*, so a ramped build
  thickens bar by bar — the lead does not. A build the drums arrive in
  arrives without its melody.
- The gap (how often the phrase speaks) is a 2-way branch on section energy;
  the phrase's *internal* rhythm (pulse count) is fixed for the whole song.
  "Moves with the song" means the energy curve, at bar resolution, should
  drive how much of the phrase is audible — the way `rhythm.thin` does for
  the drums.
- **Harmony: the lead is anchored to the song's first chord.** The motif is
  generated against `plan.events[0].chord`, and placements get a whole-phrase
  root-degree transpose. But sections now have their own progressions (peak
  cadence, release drone). A phrase re-anchored to the *first* chord of the
  song, translated by the current root, can disagree with the chord actually
  under it — and only the critic notices (at <25% agreement, by which point
  it is well too late; the only offered cure is a new take).
- **Register clash.** The lead window (G3–G5) overlaps the stab window
  (≈G2–G4) by a whole octave in the dubstep profile. A lead "riding" a drop
  is mostly riding inside the chord band. Nothing flags it.
- `LeadParams.density` is documented as "fraction of 16ths that carry a note"
  but it resolves to the *cell's* rhythm; actual song-level presence is the
  cell density divided by the gap. One quantity is wearing two meanings — the
  split is what makes "how much lead is in this song" dialable and energy-
  responsive at once.

---

## 2. The plan

Ordered so each phase lands testable. Phases 0–1 fix what is already being
built; 2–4 are the three asks; 5 hardens it.

### Phase 0 — make the lead reproduce (unblocks everything)

`compose._resolve_material`: draw `LeadParams` from the **shared** `rng`
(exactly the bass pattern — `params.bass = bassline.params_for_feel(feel,
rng, ...)`), leaving `lead_rng` to the motif alone. Both composes then start
`generate()` from a fresh stream, and a re-roll of `lead` keeps the character
and draws a new line — consistent with the sub.

- Keep the *draw ranges* as they are (0.18–0.45 density, etc.); only the
  stream changes. Values will differ from the buggy tree for any seed —
  nothing pins them yet (the tree is uncommitted).
- Tests: the three failing tests go green; add one that pins the exact
  property — `compose(params=song.params.copy())` reproduces the lead byte-for-
  byte for a spread of seeds, plus `regenerate(song, "lead")` leaves every
  other part untouched while the lead's *rhythm shape* (pulse count) is
  preserved and its degrees are not.

### Phase 1 — finish the dials

Steering surface, so the six float knobs join their two missing siblings:

1. **Choice knobs.** The `Knob` model is float-shaped; `key` already proved
   the escape hatch (a `kind` with special read/write). Add `kind="choice"`
   with a `choices` tuple and a `_CHOICES` registry in `direct`:
   `lead_style: ("call_response", "long_tones", "riff")` and
   `lead.contour: (the 8 names)`. `read_knob`/`set_knob`/`_apply_absolute`
   (validate against the table) and `diff` (compare names, not floats) get
   the two extra cases, mirroring how `mode`/`progression`/`profile` are
   handled. The page renders a choice row as a select, exactly like the key
   row; the sound panel's new lead group shows all 8 dials + 2 selects.
2. **UI: give the lead a panel.** `GROUPS` gets `["lead", "lead"]`; `PRETTY`
   gets the lead labels ("lead density", "lead syncopation", "lead span",
   "lead register", "step bias", "ornament", "lead style", "lead contour").
   Move the lead-style select out of the advanced panel into the lead group
   (one control, one path) — the advanced panel keeps what is a *full
   recompose* (profile, key, progression, seed, drum/bass details).
3. **Lead intents.** `INTENTS`: "lead_busier"/"lead_thinner" (density),
   "lead_higher"/"lead_lower" (octave), "lead_syncopated"/"lead_straighter"
   (syncopation), "more_lead"/"less_lead" (density ±, the honest presence
   dial until Phase 2 splits it). Aliases: "lead", "melody".
4. **Critic gains a lead twist.** `_lead_chord_tones` keeps the roll as last
   resort but offers first a twist toward the chord: `Op("lead.chord_lock",
   to=1.0)` once that knob exists (Phase 3). Until then, the first fix is
   `Op("lead.ornament", -0.2)` or the roll — pick the one the loop actually
   improves with; the score decides.
5. `Material.describe()` already reports the resolved lead; extend it to
   `style contour dens span oct sync` so the printed concept line and the
   readback tell the maker what the lead *is*.

### Phase 2 — the lead follows the energy curve (bar by bar)

The ask "moves with the rest of the song", in the one place it is currently
false:

- In `_build_lead`, evaluate `d = structure.density_for(
  structure.bar_energy(section, bar), "lead")` **per placement bar** and:
  - thin the placed phrase with `melody.thin_tones(phrase, d, rng)` (it
    already drops weakest metric positions first — the dub version of a
    solo), and
  - take velocity from `bar_energy(section, bar)` per bar, as the drums do,
    and
  - pick the gap from the bar's density (keep the 2-way branch; just stop
    reading the section's flat value).
- `long_tones` does the same for `per_bar`.
- **Split the quantity.** Keep `LeadParams.density` as the *cell* rhythm
  (rename the docstring to say so: "density of the phrase, in 16ths"), and
  let the energy curve own *presence* (how often and how full the phrase is
  stated). This is what makes "how much lead is in this song" a real dial
  rather than an accidental byproduct, and it is what makes a ramped build
  gain its melody bar by bar alongside the drums.
- Net effect: intro/holes get a fragment (or the cell's tail), bodies get the
  full cell, peaks get it densest, and a ramped build arrives with its lead
  exactly when the drums arrive.

Tests: in a ramped build section, per-bar lead note count is monotonically
non-decreasing across the ramp; in a flat section it is roughly constant;
the lead in a 0.2-energy section has materially fewer notes than in a
0.9-energy one.

### Phase 3 — the lead follows the harmony

- **Re-anchor per placement, don't only transpose.** For each phrase
  placement, snap the tones on metrically strong steps to the *current*
  chord with `melody.nearest_chord_degree` — the same rule `generate`
  applies, applied where the phrase actually lands. A motif derived from the
  song's first chord therefore re-homes onto the peak cadence and the
  release drone instead of ploughing through them.
- Make it a dial: `LeadParams.chord_lock: float = 0.5` — the chance a strong-
  beat tone snaps (1.0 = the lead is a counterpoint that always agrees; 0.0 =
  free line). Bounded in `direct`, rendered in the lead panel, and it is the
  critic's twist for `_lead_chord_tones`.
- **Register:** add a critic check that the lead does not mostly live inside
  the chord window (overlap in both profiles), proposing
  `Op("lead.octave", +1)` as the twist. This is the cheapest cure for the
  lead-in-the-chord-band clash and it is exactly the kind of mix knowledge
  the profile already encodes.
- Audit the octave development stages: widen the effective choice so
  `dropped`/`raised` are audible (apply the octave shift and let the window
  clamp *recentre* rather than silently fold — or move the base octave so the
  window admits both). If a stage cannot be made audible in a two-octave
  window, it should come off the ladder rather than pose.

### Phase 4 — real phrase structure

1. **Development returns home.** Replace "stage = section index" with a
   *schedule*: the ladder climbs toward the song's middle and comes back down
   to the statement for the final sections — `stage = f(index, n)`, e.g.
   `min(i, n-1-i)` mapped onto a curated 5–6 rung ladder (statement, light,
   moderate, strong, register, statement). The melody travels and comes
   home; the outro restates the opening cell (possibly displaced an octave
   down) the way records actually do.
2. **Steerable per section.** Add `development: int = -1` to `SectionSpec`
   (-1 = follow the schedule; an index picks a rung) and a `sections.<i>.
   development` entry in `_SECTION_FIELDS` (int kind, 0..10). The section
   editor gets a development select. Now "make the breakdown play the
   fragment" is one click.
3. **Close the phrase.** In the final bar of a section the lead plays, append
   a small closing figure — a 2–3 tone descent landing on the chord's
   root/third (drawn from the developed cell's tail, not new material) — the
   lead's answer to the drum fill the latest commit added. Where the section
   is a dub throw, the figure plays once into the delay and the section ends
   on silence.
4. **A two-bar cell (the bigger one, do last).** Let the motif be a 32-step
   A/A' cell: generate a second bar as a fixed development of the first
   (displaced, or answer at -1 degree), so a phrase is *two* bars with a
   relationship instead of one bar repeated. Behind the existing knob:
   `LeadParams` gains nothing user-facing at first — it is the rhythm's
   length, not a dial — but it changes what "statement" means and gives
   `call_response` and `fragment` much more to work with. Ship after 1–3
   are listening-proven.

### Phase 5 — harden and document

- Tests per phase as noted; plus a UI endpoint test that the lead knobs come
  back from `/api/knobs` with sane bounds, and one that a `lead_style`
  choice op round-trips through `apply_ops` → `compose` → `diff`.
- Critic coverage: the new register-clash and (Phase 2) "lead ignores the
  ramp" checks get findings-with-fixes like everything else, so `--fix` and
  fix-all can find and cure them.
- README: a short "the lead" section in the steering knowledge list — the
  dials, the style/contour choice, per-section development, and the rule
  that the lead follows the energy curve and re-anchors to the chord under
  it.

---

## 3. Out of scope (deliberately)

- No second lead voice / double. One melodica line is the genre's deal; the
  dub mix already gives it its space.
- No per-note melodic editing in the UI. The model is motif-plus-knobs;
  hand-placing notes would fight the re-roll/trail machinery.
- No new profiles. The lead changes in the two existing families (dub lead
  enters at versions; dubstep lead rides drops) fall out of Phases 2–3 and
  the existing mute maps.
- Lead *timbre* is a Deluge/UI concern (the page's synth is already
  melodica-ish); this plan is about what the engine writes, not how it sounds
  in the browser.

## 4. Verification

After each phase, the listening test matters as much as the suite:

1. `riddim-cli compose --profile dubstep --seed N --check` — the critic's
   lead findings should dry up across a batch
   (`riddim-cli batch --profile dubstep --count 20 --top 5 --fix`).
2. In the UI: move each of the eight lead dials and hear exactly one thing
   move; the changed-bar chip should say `lead density 0.35→0.55` and the
   piano roll should shade exactly the lead.
3. The structural ear-check: a ramped build — does the lead arrive *with* the
   thickening? A peak — does the line re-anchor to the cadence? The outro —
   does it end on the opening cell?

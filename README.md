# riddim

Theory-grounded dub and dubstep composition for the Synthstrom Deluge and a
modular rack.

This is a deterministic music-theory engine, not a model. Nothing is learned
from a dataset; every note comes from a rule you can read, argue with and
change. That is the point — when a bassline is wrong you can find the line
that made it wrong.

## Show prep: from nothing to a set

```bash
# 1. Compose a show: 48 tracks across the six feels, each the best of 10
#    candidates, revised by the critic, ordered as a set, with audio previews
./riddim-cli corpus --count 48 --candidates 10 --audio --out show

# 2. Audition: open show/setlist.md, play the .m4a previews, keep favourites

# 3. Mould a favourite: variations that keep it itself
./riddim-cli vary --params show/07_tower_archive.json --how hook   --count 6 --out v
./riddim-cli vary --params show/07_tower_archive.json --how mutate --count 6 --out v
#    --how: mutate | hook | groove | arrangement | structure | drums | bass | wild

# 4. Steer one by hand in the UI (every knob below, plus words: "catchier",
#    "bigger drop", "tighter", "mutate hook"...)
./riddim-cli ui

# 5. Put it on the Deluge - the .json rebuilds the exact song
./riddim-cli send --params show/07_tower_archive.json --mode song
```

Every track in a corpus is written as `.mid` (audition in any DAW), `.json`
(its complete params — `send`, `compose`, `render` and `vary` all take
`--params`), `.txt` (key, tempo, structure, hook, player notes) and, with
`--audio`, an `.m4a` preview. `setlist.md` / `setlist.csv` order the set:
tempo blocks from 72bpm roots through steppers and half-time into 140, keys
chained a step at a time round the circle of fifths (Camelot codes printed),
each block climbing in energy.

**Kit rows.** The drum kit now uses two more rows: **42 crash** (the impact
on every drop and phrase) and **43 riser** (a noise sweep or reverse cymbal,
one long note through the end of each build). Load sounds there, or pass
`send --no-fx-rows` to leave them out.

## What makes the music work

The brief: parts that talk to each other, complete structure and dynamics,
tension that builds, clear roles, and real hooks. Each is a module:

| | Where | The idea |
|---|---|---|
| **The hook** | `theory/hook.py` | Built from a **motive**: a rhythmic idiom (dotted 3-3-2, stabs, pickups, long-short, offbeat, three-against-four, flowing, syncopated) with a pitch shape, stated again and again across a unit and moved along a contour — the riff's identity. Four units in a real form — *sentence* (a a′ b c, the b climbing to the one high point), *period* (a x a y), *riff* (a a a a′), *call-and-answer* (a — a′ —); units end open until the last closes; a breath in every unit; the mode's colour tone once. 32 candidates are fitted to the real chords and scored on soundness **and on matching the dials** — so a dial moves the hook, rather than the selector averaging it away. |
| **The performance** | `compose._build_lead` | How the lead plays the hook: gate (staccato riff ↔ legato song), slides (overlapping notes, so a mono synth glides), accents, grace notes flicked in from below, octave flicks, trills and mordents, and dub echoes printed into the breaths. Each decision is seeded by the note's position, so changing the hook does not reshuffle how the untouched notes are played. |
| **The theme plan** | `arrange/interplay.py` | What the hook does in each section: a quiet tease at the end of the intro; its head climbing a step at a time through the build and doubling in speed into the drop; the whole hook, literal, in the first drop (decorated, then lifted an octave, in later drops); the first half at half speed over a breakdown. |
| **Kinds of lead** | `theory/hook.PHRASINGS`, `compose.lead_params_for` | Each song draws what *kind* of lead it has - **call** (figure, held note, space), **riff** (an ostinato), **song** (a singable line of mixed note values), **stabs** (one or two punchy hits, home-leap-home-leap), **drone** (two or three long notes a unit), **arp** (chord tones rolling), **chop** (a garage vocal chop: a stuttered syllable, a step, a clipped vowel) - and each is played its own way (stabs short and hard with octave flicks, song and drone legato and sliding). `hook.phrasing` is a dial; words: `arp`, `stabby`, `singing`, `drone`, `ostinato`. |
| **Phrasing** | `theory/hook.py` | A unit is a *call*: one statement of the motive by default (density adds a second or third), each a quick figure that lands on a **held note** — the note a listener sings — then the answer space. No back-to-back blips. |
| **Call and reply** | `arrange/interplay._respond`, `compose._sub_reply` | Every answer space is handed to a responder, bass and chords taking turns: the **sub** plays the call's figure back in bass moves (root, third, fifth, octave, b7 below), the **chords** stab it with the voicing lifted where it rises; their ordinary pattern steps aside in the window so the reply is heard, and the lead's printed echo keeps out of answered space. In the call-and-response style the call holds through its own breath and the whole next unit is the reply. `interplay.answer` sets how many calls get one. |
| **The conversation** | `arrange/interplay.py`, `compose.py` | The sub doubles the hook's rhythm (strong beats on the root, the rest in unison with the tune) on the passes the lead sits out, and sometimes under it; chord stabs duck out of the lead's onsets and answer its breaths with the hook's own rhythm; the bass picks up into the next call; the pad's top voice is a guide-tone counter-line moving against the tune. |
| **Room for each part** | `compose._yield_to_lead`, `_build_chords`, `_build_pad` | Complementary rhythm: in a beat where the lead plays a figure, the bass keeps only its note on the beat and holds it, and the skank keeps one stab; they move again when the lead holds or rests. When pad and stabs play together the pad drops to a single moving voice instead of doubling the chord. All scaled by `interplay.space`. |
| **Variety between songs** | `groove.FEELS`, `profile.bpm_range`, `structure._phases` | Each song draws its tempo from the genre's range, its drum feel from the whole family (one-drop, flyer, pickup, rockers; steppers, skip; half-time, push, double-kick, tresillo, sparse, roll), its skank from the named library or the generator (`stab_pattern` is a dial), and its shape from a grammar that now includes groove sections. |
| **On the grid** | everywhere | Every note is written dead on the grid - no swing, no timing humanisation, no laid-back snare (velocity still varies). Feel is set on the Deluge, whose swing moves every clip together; swing printed into the notes would fight it. |
| **Idiomatic kit** | `groove.drums` | Hats come from a drummer's ladder of figures - offbeat chick, eighths, eighths with a skip, running sixteenths (half-time stops at the skip) - chosen by `hat_density`, accented on or off the beat by `hat_offbeat_bias`. Ghost snares and rims take the positions a player reaches for, not random sixteenths. The shaker sits on the offbeat eighths. The one-drop's empty beat 1 is never filled by a locked kick. |
| **The rhythm section locks** | `compose._lock_kick`, `compose._on_eighths` | The bass is written first and the drums follow it: the genre's anchor kicks stay (one-drop on 3, half-time on 1, steppers on the beat) and every other kick lands on a bass note. The sub plays on the eighth-note grid only - a sub on the in-between sixteenths, swung late, is heard as out of time. The groove repeats bar for bar - each drum voice keeps one pattern per section (varied only on the A′ bar, fills and builds), and the kick locks to where the bass *usually* plays in the section, not to every reply - because a kit that reshuffles every bar is heard as out of time. A fill never talks over the lead singing into the next phrase. Fills are short: a beat of sixteenths into a drop, every sixteen bars at most inside a section. |
| **Songs that develop** | `compose._approach`, `compose._layers`, `hook.contrast_params` | Builds end on a tension chord - the dominant with its leading tone, or the phrygian bII - so the drop's downbeat resolves (`dynamics.cadence`). Long drops unfold by eight-bar block: hats and shaker held back at first, a stripped block of drums, bass and tune, then everything (`dynamics.layering`). Later drops bring a **B-theme** - the hook's rhythm and kind, contour turned round, and the opposite character (a busy hook gets a singing second theme, a sparse one a busier answer), starting a third away if it would open on the hook's note - for one pass before the hook returns (`hook.contrast`). Intros arrive in layers (the beat lands halfway); outros strip to the foundation. |
| **Development, not repetition** | `harmony.develop`, `compose._bass_turnaround`, `_build_chords`, `interplay._sequenced` | Phrases have ends: bar 4 of every eight bends the bass figure and the stabs anticipate the coming chord on the & of 4; bar 8 turns the bass around (fifth, seventh, a step into the next root) and leaves the last beat to the drums. In the second half of a long drop the stab voicing lifts an inversion. Later drops *return and develop*: the first drop's loop, then the loop with one or two chords bent (a third-related substitute, the dominant where it went home, the deceptive bVI) - each later drop differently - with the harmonic rhythm quickening if the developed cycle would not fit; and one pass in four of the hook is sequenced, its restatement climbing a third. The last drop can **lift** into a new key (`dynamics.lift`; words `key lift`, `no lift`): up a tone or a semitone, from the approach bars - which become the new key's dominant - to the end, every pitched part together. A mid-song **breakdown is the bridge**: it gets its own progression, one the drop and the riddim never use, while intro and outro keep the rest chord so the song opens and closes on the same ground. **Phrase dynamics**: hats, shaker, ghost snares and chord stabs breathe with the eight-bar phrase - lean on its first downbeat, ease off, push into the turnaround, the hats opening up as it goes - while kick, backbeat and sub never move. The pad's moving top voice **suspends** across chord changes - the old note held into the new chord and resolved down a step a beat or two later, the 4-3 and 9-8 of every chorale (how often: `interplay.counter`). A tune may sing a chord's **available tensions** - the ninth, the eleventh over a minor chord, the sixth, the seventh - which the hook auditions, the listener and the critic all count as agreeing (`harmony.sits_on`), but never a note a semitone against the chord's third or fifth. **Dub throws**: in a dub-out - band stripped to drums and bass - a lone skank is let through every couple of bars, short and hard on an offbeat, so the chords' delay send fills the bar; at the end of each eight a scrap of the hook's head on the lead. The section then features what it throws (the Deluge gets a clip for it) while the song's recipe keeps the dub-out bare. **Hats at three levels**: the accent, the other eighths at a medium touch, the in-between sixteenths ghosted (the garage skip just audible). **Phrase-final lengthening**: the note a figure lands on - the one a breath follows - is held out however short `lead.gate` clips the run before it (stab leads excepted), so a staccato riff arrives instead of stopping. **The lead's register is a melody's register**: it sits above what the chords actually play (not the top of their window, which put tunes a fourth too high), may dip beside the stabs for a few notes, and is kept from living more than an octave above them; octave-up leads are drawn rarely. **Builds approach the climax without spending it**: the climbing fragment stops a step below the hook's high point (starting lower where the figure already sits high), and builds play no octave flicks. **The last drop is the whole band**: with two or more drops, the pad joins the last one - the only place all six parts play - and it arrives whole, holding nothing back in its first block. **The coda remembers the tune**: the outro's lead plays the hook's cell, then just its head, softer, and is silent for the last bars; the final hit lands in the song's key (its new key, after a lift). **The bass states every chord change**: a reply carried over the barline never takes the next bar's downbeat, and a one-drop bass that skips the one lands a new chord's root with the kick on beat 3. **Builds tighten**: over the last bars of a ramped build the hats climb the ladder a rung, then two, under the snare roll, until the floor drops out of the last bar. **Call and answer**: where the chords answer the lead they play its figure, say it again quieter as a call-back where the breath is long, and then the skank comes back - the answer takes only the room it needs. **The first drop establishes the hook** - it sings every pass (all but one in a long drop); later drops may rest it and hand it to the bass. A garage chop comes round every bar. The hook sings **over** the bass, not on it: on strong beats it takes the chord's third, fifth or seventh and keeps the root for its cadence. |
| **A set, not one song** | `harmony.PROGRESSIONS`, `compose._resolve_scale` | Keys come from all twelve tonics (the tonic folds into the sub register, so every key fits) - more keys for the set ordering's harmonic mixing too. The progression library carries each mode's real vocabulary - dorian's i-IV-bVII, i-bVII-IV, the climbing i-ii-bIII-IV; minor's anthem i-bVI-bIII-bVII and the circle i-iv-bVII-bIII; harmonic minor's i-iv-V and i-V-bVI-V - so no one drop progression takes more than a quarter of a genre's songs. The chop's stutter follows `hook.repeat`: none, the double hit, a triple. |
| **Kinds of bassline** | `compose._bass_style` | Each song draws a bass style - **walk** (a melodic line), **sustain** (one big held note a bar, the dubstep drop sub), **pulse** (the root on every beat, steppers), **bounce** (root and octave in eighths), **sparse** (strong beats, held), **skip** (the 2-step bass: root with the kick on 1, an octave pop on the & of 2, the root again with the late kick, a pickup every other bar) - weighted by genre. `bass.style` is a dial; words: `held bass`, `walking bass`, `pulsing bass`, `bouncing bass`, `sparse bass`, `skippy bass`. |
| **UK garage** | `groove.TWO_STEP…`, `profile.UK_GARAGE`, `DARK_GARAGE` | Dubstep's other parent. Kits have an **idiom** - `roots` or `garage`. The garage kit: 2-step kicks (on 1 and the & of 3, or the late sixteenth after it, never on beats 2-4 - extra kicks that would land there are refused), hats that skip in pairs into the backbeat (and never climb to running sixteenths, where the skip disappears), ghost snares pushing into the one, rimshots on the in-between sixteenths, a running sixteenth shaker for the Deluge's swing to shuffle. **Minor-ninth** chords (`chord_size` 5: major ninths only - never a flat nine - and a maj7 degree keeps its ninth as an add9), organ stabs that dance round the kick, the skip bassline, and the **chop** lead - a stuttered syllable, a step, a clipped vowel, a sampler cutting up a vocal. Garage arrangements stop more than they build: shorter rolls, more mid-drop switch-ups, the chords playing almost throughout. `uk_garage` (130-136) is the whole idiom; `dark_garage` (134-140) is the proto-dubstep record - 2-step kit, deep sub, dark minor and phrygian harmony, dubstep's arrangement. And dubstep inherits: its pool includes half-time kits with the 2-step's late kick and skipping hats (`half_time_skip`, `half_time_garage`), organ stabs, ninths, the skip bass and the chop. Words: `garage` / `ukg` (garage the song up in its own meter: the 2-step kit or its half-time cousin, skip bass, ninths, organ stabs, chop lead), `2-step`, `skippy`, `4x4`, `ninths`, `organ`, `vocal chop`. `drum_feel` is a select over every named feel. |
| **Tension & release** | `arrange/dynamics.py` | Builds roll — snare in quarters, eighths, sixteenths, thirty-seconds — over a riser; the floor drops out of the last bar; the band stops dead for the drop gap; the drop lands on a crash; long drops break for half a bar at their phrase ends; drops escalate so the last is the biggest; the whole song ends on a hit. |
| **Roles & register** | `compose._theme` | The hook is placed as a block (its leap intact) just above the chord window; chords voice rootless over the bass; maj7s stay triads so the tune never grinds on the root. |
| **Harmony that serves the hook** | `theory/harmony.realise_cycle` | Where the hook sings, the chord cycle is laid to the phrase length, so every pass meets the same chords and comes back note for note. Borrowed chords lend their tones: over a major V the tune raises its seventh with the chord. |
| **The listener** | `analyze.py` | Hook strength, chord fit, clashes, register, space, answers, contrast, arc, tension, length, hook presence — one 0..1 score that ranks a batch. Across 64 songs it averages ~0.92 (the original engine scored ~0.72 on the same measure). |

### The dials

Every one of these is a knob in the UI and a path for `Op` / the trail /
the critic's fixes:

| group | knobs |
|---|---|
| `hook` (what is sung) | `rhythm` (idiom) · `form` · `contour` · `density` · `syncopation` · `motion` (steps ↔ leaps) · `repeat` (repeated notes / re-strikes) · `leap` · `span` · `climax` · `repetition` · `colour` · `outside` (pentatonic ↔ whole mode) · `breath` · `unit_bars` · `candidates` |
| `lead` (how it is played) | `lead_style` (hook · call_response · riff · long_tones) · `octave` · `gate` · `slide` · `accent` · `grace` · `flick` · `echo` · `ornament` |
| `interplay` | `presence` (how much the lead sings) · `answer` (fill its gaps) · `lock` (sub takes the hook's rhythm) · `counter` (pad counter-line) · `space` (stabs duck the lead) |
| `dynamics` | `roll` · `riser` · `drop_gap` (beats) · `impact` · `switch_up` · `fills` |
| harmony | `tension` · `bars_per_chord` · `chord_size` (3 triads / 4 sevenths) · key · mode |

`tests/test_show.py::test_every_lead_dial_moves_the_lead_its_own_way` holds
every lead and hook dial to that: low to high, each must move its own audible
feature (note count, syncopation, interval size, repeats, range, length,
accent contrast, glides, pitch) in its own direction.

Words resolve to moves on them: `punchier`, `smoother`, `staccato`,
`legato`, `slidier`, `more echo`, `flicks`, `grace notes`, `trills`,
`leapier`, `steppier`, `more repeats`, `dotted`, `stabs`, `pickup`,
`triplet`, `offbeat`, `flowing`, `catchier`, `wider hook`, `riff`,
`call response`, `long tones`, `more hook`, `chattier`, `tighter`,
`more counter`, `bigger drop`, `more build`, `more breaks`, `sevenths`... and
**mutation** — `mutate`, `mutate hook`, `mutate groove`, `mutate arrangement`,
`wild` — is a random twist across a group of knobs with every seed frozen: the
song moves somewhere new and stays itself, on the trail, undoable. **New
hook** (UI button, or `vary --how hook`) redraws only the hook's seed: a new
tune over the same band, and every part that answers it follows.

## Writing like a person

Nobody hears a generator in a wrong note. They hear it in *too many* notes:
a hook that walks the whole scale, a sub that visits every degree, a riff
whose ornaments land somewhere new on every pass. People write with a
handful of pitches and say the same thing again. `riddim/tells.py` measures
exactly that, on the first drop, against the range each family's records
sit in:

| tell | what it is | dub | dubstep | garage |
|---|---|---|---|---|
| `lead_vocab` | distinct pitch classes in the lead | ≤ 6 | ≤ 5 | ≤ 5 |
| `lead_range` | semitones, octave flicks folded | ≤ 12 | ≤ 12 | ≤ 12 |
| `lead_density` | notes per bar | ≤ 5 | ≤ 4 | ≤ 6 |
| `lead_restated` | sung bars that repeat one 1/2/4/8 bars back | ≥ 0.45 | ≥ 0.5 | ≥ 0.45 |
| `drop_chords` | distinct chords under the drop | ≤ 4 | ≤ 2 | ≤ 4 |
| `sub_extra` | sub pitches beyond the chords' roots and fifths | ≤ 3 | ≤ 2 | ≤ 2 |
| `sub_on_root` | share of sounding sub time on the chord root | ≥ .45 | ≥ .75 | ≥ .55 |

```bash
./riddim-cli tells                       # 16 songs of each family
./riddim-cli tells --profile dubstep --count 40
```

The ranges are genre judgement written down to be argued with; calibrate
them against reference MIDI when it is to hand. What answers them:

- **The pentatonic palette** (`hook.PALETTES`). Every hook is built inside
  its mode's five-note pentatonic, with the mode's colour tone placed once
  on purpose; `hook.outside` is the chance a stray scale tone is let
  through (drawn 0–0.3; words `pentatonic`, `outside`, `inside`). The hook
  scorer rewards **economy** - three notes for a stab or chop, four for a
  call or riff, five for a sung line - where it used to reward variety.
- **Narrower spans.** Most hooks live inside a fifth or a sixth; garage and
  140 hooks are drawn tighter still.
- **A riff is played the same way every time.** Flicks, graces, trills and
  accents are seeded by the note's place *in the hook*, not in the song, so
  each pass is performed identically - this drop and the next. A flick only
  pops a note up an octave if it stays within an octave of its neighbours.
- **The 140 drop.** `Profile.drop_chords` caps the drop's loop: two chords
  for dubstep and dark garage (a drone, i-bII, i-bVI, i-V), four elsewhere.
  The dubstep-family drop sub keeps to the chord's root and fifth (and
  phrygian's b2), a walking sub to the chord's four notes, with one approach
  note into the next phrase - everything else is snapped, keeping its rhythm.

Across 24 fresh seeds a family, the giveaways per song went from 2.0-4.1
to 1.0-2.0 (dubstep 4.08 → 1.92), every family's hook range from 14-18
semitones to 10-12, and restated drop bars from about a fifth to about a
half. `tests/test_tells.py` holds a batch of each family to the ceiling.

## Ears: listening against real records

The engine cannot hear, so it measures. Three tools, and a reference set:

- `ml/collect_refs.py` collects reference recordings (archive.org items or
  anything yt-dlp reads) into `references/` with a manifest of where each
  came from and under what licence - kept out of git. Run it on your own
  machine.
- `riddim/listen.py` turns any recording into numbers and pictures: tempo,
  swing, band balance, stereo width, texture, the energy curve, and a
  spectrogram with a two-bar close-up on the sixteenth grid.
- `ml/judge.py` embeds recordings with CLAP and scores a batch of renders
  against the references: **FAD** (Frechet Audio Distance, the standard
  measure for generated audio), each track's **closeness** to the reference
  centroid, and a zero-shot **real-record vs machine** probability. It is
  also the probe that says *what a passage sounds like* - "dubstep", "new
  age", "video game music".

What it found on the 29-track DMZ catalogue, and what changed because of it:

| finding | change |
|---|---|
| the preview sounded like a MIDI file (0.000 "real") | the renderer rebuilt: stereo, the sampled Pearl kit with round-robin, the EMT-140 plate, a drum room, a hall, a tape-echo dub delay that runs up into drops, a mid-bass growl at 140, kick pumping, bus glue and tape |
| DMZ intros keep the mids and snare within ~4 dB of the body; sub and kick are 20-50 dB down until bar 16-24 | `StructureParams.low_end_out` for the dubstep family: intros and breakdowns are the band without its low end; the first drop mostly comes straight out of the intro |
| a 16-bar pad-only breakdown is unlike anything in the catalogue | breakdowns keep percussion, chords and the tune |
| the records are dense with hand percussion | `theory/percussion.py`: congas, bongo, clave, tambourine and tom runs on the perc kit (rows 44-50) |
| the lead riding on top of the mix is the loudest machine tell (0.12 → 0.64 "real" with it 10 dB down); a pad in a big hall reads as new age | the lead sits in the mix (`audio.LEAD_LEVEL`), the pad darker and drier |
| a climbing lead fragment and an EDM ramp into a 140 drop read as machine | dubstep-family builds are a held level with no lead tease; smaller rolls and risers |

Across a fixed panel of six songs the renders went from FAD 0.54 to 0.30
(the DMZ catalogue against itself: 0.12), closeness 0.71 to 0.83 (DMZ median
0.84) and "real record" 0.00 to 0.41 (DMZ median 0.92). The drops and the
breakdowns now pass about half the time; the tells left are in the builds
of the non-140 profiles and the melodic parts.

**Kit rows (perc kit).** The hand percussion uses rows 44 conga low, 45
conga high, 46 bongo, 47 woodblock, 48 tambourine, 49 tom low, 50 tom high.

## Why it is built this way

Four things are hard about writing this music, and each maps to one module:

| The problem | Where it is solved | The idea |
|---|---|---|
| Finding a melody | `theory/melody.py` | Invent *one bar*, then develop it. Transpose, invert, fragment, displace. Development is mechanical; invention is not. |
| Finding a rhythm | `theory/rhythm.py` | Metric weight, Euclidean distribution, and starting from a full grid and *removing* — because in dub the pattern is the holes. |
| Structuring a song | `arrange/structure.py` | A dub arrangement is subtractive. One riddim, and sections are mute maps plus an energy number. |
| Chords that work | `theory/harmony.py`, `theory/voicing.py` | Modal vamps, not functional harmony. Voice leading picks inversions that move least. |

Everything in `theory/` is pure — no MIDI, no hardware, no I/O — so it is
testable, and the tests check musical facts rather than that the code runs.

## Quick start

Use the `riddim-cli` launcher in the repo root — it works from any directory:

```bash
# Write a MIDI file you can drop straight into a DAW
./riddim-cli compose --profile dubstep --key "G phrygian" --out sketch.mid

# See what would land on the Deluge, without sending
./riddim-cli plan --profile dub --seed 42

# Run the engine's own knowledge in reverse over the song: which sections are
# not doing what they claim, which notes are out of register, where the energy
# curve lies
./riddim-cli compose --profile dubstep --seed 40 --check

# Same, but let the critic fix what it finds: apply its fixes until the song
# is clean or a step stops helping, printing the trail
./riddim-cli compose --profile dubstep --seed 40 --check --fix

# Compose a batch, let the critic rank it, keep the best one
./riddim-cli batch --profile dubstep --count 30 --top 5 --fix --out best.mid

# Send it into the Deluge session, ready to jam
./riddim-cli send --profile dub --seed 42

# Or make the arrangement play itself start to finish
./riddim-cli send --profile dub --seed 42 --mode song

# Print the whole theory library: progressions, bass figures, riddims, claves
./riddim-cli library

# Hear it without the rig: a WAV preview of any song (numpy + scipy)
./riddim-cli render --profile dubstep --seed 7 --out preview.wav
```

`python -m riddim ...` also works, but **only from the repo root** — from
inside `riddim/` it fails with `No module named riddim`, because that is where
the package needs to be importable *from*. The launcher resolves its own
location, so it does not care where you are.

Same `--seed` always gives the same song. Omit it and you get a new one.

## The rig it assumes

Deluge MIDI out into the Coral, drums on a Deluge kit:

| Role | Goes to | Notes |
|---|---|---|
| `sub` | MIDI ch 1 → rack | Single oscillator. Forced monophonic — no two notes ever overlap. |
| `chords` | MIDI ch 2 → rack | Short offbeat skank stabs. Put this channel on the Nostalgia send. |
| `lead` | MIDI ch 3 → rack | |
| `pad` | MIDI ch 4 → rack | Long held chords under the stabs. |
| `drums`, `perc` | Deluge kits | Two separate kits. Kit row 36 is the bottom row. |

Every pitched part leaves the Deluge. One consequence: the firmware only
accepts param automation on **synth** clips, so with nothing pitched staying
internal, **no filter automation is written at all**. That is correct — the
filters being moved belong to the rack — but it means all filter movement is
played rather than sequenced, and the energy curve lives in note density and
the section mutes instead. `--all-deluge` keeps parts internal and does get
automation, which is handy for sketching with the rack unpatched.

`--all-deluge` keeps everything internal if the rack is not patched.

## Jam mode vs song mode

Each section of the arrangement becomes a Deluge **section** (colour group),
holding one clip per role that plays in it. Hitting a section pad launches the
whole band for those bars.

The `--mode` flag sets the section repeat counts, which is what decides how it
behaves under your hands:

- **`jam`** (default) — every section set to infinite repeats. A section loops
  until you launch another. You drive the arrangement: hold on `riddim_a` for
  as long as it is working, drop to `dub_out` when you want the hole.
- **`song`** — every section plays once and hands over to the next, wrapping
  at the end. The arrangement plays itself.

This matters because the firmware only auto-advances when a section's repeat
count is >= 1 (`session.cpp`, `armNextSection`), so the two modes are one
number apart and behave completely differently.

The channel map lives in `song.py::Rig` — edit it there, it is one dataclass.

## Live sets

A record and a live set want different shapes. `--live` (or saying "live"
in the UI) redraws a song's structure as a palette of short loops for jam
mode - the same hook, key, harmony and groove:

| section | bars | what it is |
|---|---|---|
| `groove` | 8 | the entry: drums and bass from bar one, the riddim at full groove |
| `build` | 8 | into the drop |
| `drop` | 16 | |
| `break` | 8 | the *only* low point (dub-out or breakdown) - it plays when you launch it |
| `build_2` | 8 | |
| `drop_2` | 16 | the bigger drop, the whole band |
| `exit` | 8 | a groove to leave on, not a fade |

`./riddim-cli ui --live` makes every take, re-roll and redrawn structure in the
UI a live track.

No long intro, no outro fade: in jam mode every section loops until you
launch the next, so you hold energy as long as you like and bring it down
only when you mean to.

### Putting a set on the Deluge, unattended

With the custom firmware's `SAVE_SONG` / `LOAD_SONG` commands (AI SysEx
19 and 20 - branch `ai-save-load` of the firmware fork), the whole set goes
onto the card with one command:

```bash
# once, on the Deluge: a song holding your instruments with a template clip
# per role (rows as in --template-rows), saved as SONGS/RIDDIM TEMPLATE.XML

# compose a 12-track live set and save every track to the card
./riddim-cli set --count 12

# or take a corpus you already like, as live tracks
./riddim-cli set --dir show --live --prefix "SET1 "
```

For each track it loads the template, writes the track's clips into it,
arms the track's first section and parks the template's empty clips out of the
way (firmware `ARM_SECTION`, 21) - so a loaded song starts its music the moment
you press play - and saves it as `SONGS/<NN title>.XML`, then moves on. The transport must be
stopped and the Deluge on a main view (not in a menu) - otherwise the
firmware answers `BUSY` and the command says so. `--overwrite` replaces
songs of the same name; `--dry-run` walks the set without a Deluge.

### Saving from the UI

After **→ deluge**, **💾 save** saves the song on the Deluge as
`<session> NN <title>` (e.g. `FRI 03 Iron Frequency`). The session box
(default: today's date, `MMDD`) keeps a night's songs together on the card,
and the number keeps them in the order you made them - the Deluge lists songs
alphabetically. The counter lives in `~/.config/riddim/sessions.json`, so it
carries on after a restart; a name already on the card takes the next number,
never an overwrite. Needs the save/load firmware (`ai-save-load`).

## Template songs

**The Deluge has no SysEx command for setting a MIDI clip's output channel.**
A MIDI instrument is identified by its `(channel, channelSuffix)` slot, and
the firmware's own channel-change path parks the instrument on `-1`, hunts for
a free slot, and refuses with "no free channel slots" if there isn't one — so
a naive "set channel" command would let two instruments collide on a slot the
rest of the firmware assumes is unique.

The way round it is a **template song**, which needs no firmware change and
solves more besides. Cloning a clip shares the source's *output*, so a
generated clip cloned from a template clip inherits that instrument entirely —
its MIDI channel, its patch, its sends.

Set up one song by hand with a clip per role, each on the right channel and
patch, and save it. Then:

```bash
# Rows as you see them: row 1 is the TOP clip on screen.
./riddim-cli send --profile dubstep --seed 1312 \
    --template-rows "sub=1,chords=2,lead=3,pad=4,drums=5,perc=6"
```

> **The Deluge displays clips in reverse of their internal order.**
> `visibleClipIndexToInternal` in the firmware is literally
> `numClips - 1 - visibleIndex`, so the clip at the top of the screen is the
> *last* one in the list. Use `--template-rows` and count off the screen;
> `--template-from` takes raw internal indices and is easy to get backwards,
> which points every role at the wrong instrument — invisible until the
> bassline comes out of the chord patch.

Every generated clip clones from those, so the role-to-channel map in `Rig`
describes something real instead of something hoped for. The template clips
are only ever clone *sources* — never selected, cleared or written to — and
the generated clips are appended after them.

Beyond channels, the template also fixes the Coral patch per channel, the
Deluge pad preset, and any mixer/send routing you want the Nostalgia fed from.

`--start-index` defaults to the end of the current song, so sends append.
Passing one past the end is rejected rather than silently misaligning
everything: the firmware clamps an out-of-range insert to the end of the list,
which would put the clip somewhere other than where the index counter thinks.

## Model hooks: a trained model writes the drops' lead

riddim keeps the structure, the band and the Deluge plumbing; a trained
model writes the hook. The model is the Anticipatory Music Transformer
(`stanford-crfm/music-medium-800k`, Apache-2.0), fine-tuned on the open
Lakh MIDI files (CC-BY) that *play* most like early dubstep and dub - picked
by content, not by tags: half-time and one-drop drums, offbeat skanks, a
sparse sub, minor modes. It hears riddim's band as it will be played and
writes the lead over it:

* it drafts several four-bar phrases over the first drop, and riddim keeps
  the one that works best as a hook (on the chords where it lands on the
  beat, a tight register, small leaps, a rhythm that repeats inside itself);
* riddim arranges the drop around it - hook, hook, answer, hook - and the
  model writes the answer having heard the hook twice;
* later drops bring the same hook back, bent onto their own chords where they
  differ, with fresh answers.

Two styles: `rhythm` (default) - the model picks the pitches on riddim's own
lead rhythm; `free` - the model writes the rhythm too.

```bash
riddim-cli compose --profile dubstep --live --model-lead --out song.mid
riddim-cli render  --profile dubstep --seed 11 --model-lead free --take 2
riddim-cli set --count 12 --model-lead          # a live set with model hooks
```

In the UI, **🤖 model hook** (shift-click: free) writes the drops' lead; each
press is the model's next take, undo brings riddim's lead back, and a twist
recomposes (press again to hear the model over the new band).

The model runs in its own Python environment, so riddim needs nothing new:
riddim starts `ml/lead_worker.py` with `RIDDIM_ML_PYTHON` (default
`~/ml/venv/bin/python`) and the model in `RIDDIM_ML_MODEL` (default: the
newest fine-tune under `~/ml/models/ft_dmz`, else the base model). The
training pipeline - data, the content filter, the fine-tune, benchmarks - is
in [ml/README.md](ml/README.md).

## What it deliberately does not do

Patch design, delay throws and filter performance are left to you. The tool
writes filter automation as a *floor*, marks which sections want a delay throw,
and then gets out of the way. The Nostalgia and Pam's are instruments, not
render targets.

## The engine is generative, not a preset browser

Nothing on the default path is a lookup. Each musical decision is made by a
generator with musical parameters:

| Decision | Generator | Its knobs |
|---|---|---|
| Drum feel | `theory/groove.py` | kick anchors, kick density and syncopation, snare ghosts, hat density and offbeat bias — one-drop, steppers and half-time are *points* in this space, not the only points |
| Chord-stab rhythm | `theory/groove.py` | density per felt beat, syncopation; never on the downbeat |
| Progression | `theory/harmony.py` | a root-motion grammar over the mode's stable triads, scored for tension; names its output by analysis (`i-bVII-bVI-V`). One per section: body sections share the tune's core, every peak shares one cadence, every hole shares one rest chord |
| Bassline | `theory/bassline.py` | density, syncopation, approach tones, octave jumps, fifth weight, contour, anticipation; phrasing as *operations* — state, restate, answer |
| Structure | `arrange/structure.py` | an energy grammar (intro? build peak [release build peak]* outro), roles entering subtractively by threshold, deliberate holes. Builds *ramp*: the section's energy climbs bar by bar toward its level, so the approach to a peak is audible |
| Drum fills | `theory/fill.py` | the last bar of a phrase, specified by the energy of the section ending: soft rims out of a hole, a climbing snare run out of a build, sixteenths and a hard thwack out of a peak. All of it in the second half of the bar — a fill points *at* the next downbeat, never pre-empts it |
| Melody | `theory/melody.py` | contour, span, chord-tone snapping, then development transforms |

The old named figures, riddims and templates still exist in the library and
can be asked for by name, but a seed never touches them.

## Params: every knob, one object, a seed per part

`riddim/params.py` holds a `SongParams` with every parameter above. Anything
left unset is drawn from the seed and **written back**, so after one
`compose()` the params describe the song completely — feed them back and you
get byte-identical notes (there is a test for this). Each part, and the
structure and harmony, has its **own seed**, so regenerating the bass is a new
bass seed and nothing else moves.

```python
from riddim.params import SongParams
from riddim.compose import compose, regenerate

song = compose(params=SongParams(profile="dubstep_140", seed=77, key="D phrygian"))
song.params.bass.approach = 0.8          # more passing tones
song = compose(params=song.params)       # only the bass changed
song = regenerate(song, "drums")         # new drums, everything else identical
```

Sections carry two extra fields the UI can set: `progression` (a library name
like `andalusian`, a Roman-numeral string like `i-bVII-bVI`, or the resolved
name of a generated progression; `None` inherits the tune's core) and `ramp`
(0..1, how much the section climbs through its own span). Both resolve, write
back, and reproduce, exactly like every other knob.

## The self-critic

Generation is one direction of the engine; `riddim/critique.py` is the other.
The same knowledge that keeps the generators musical runs in reverse over a
finished song and says plainly where it is not doing its job: a part listed in
a section but silent there, notes outside a register window, a bass that never
puts a root under a fast-changing chord, a peak that is not the loudest thing,
a build that never climbs, a hole that is loud. It also knows the show rules: a hook too weak to
be one (re-roll the hook), a peak nobody carries the hook in, a lead buried
in the stabs' register, a drop with no roll, gap or impact.

It reports, and it *proposes*. Every finding carries the change that would
cure it, in the steering vocabulary: a **twist** for what a knob can fix (the
harmonic rhythm, a section's energy or ramp) and a **re-roll** for what only
a new take can (a silent part, a note outside its window). `--fix` runs the
revision loop — worst finding, apply its fix, recompose, re-critique — until
the song is clean or a step stops improving its score, and prints the trail,
rejected steps included. A twist keeps every seed frozen, so a fix moves the
song a little; a step that would not improve it is not kept. The UI has the
same loop as a check panel with a fix button per finding, and **fix all**.

## The UI

```bash
./riddim-cli ui        # opens http://127.0.0.1:8765
```

Standard library only. The page shows the structure as an editable timeline
(add, remove, reorder, resize, set energy, toggle which roles play, name a
section's own progression, set how hard it ramps), the steering surface, a
piano roll per part with the chord tones shaded, a check panel with the
critic's findings and a fix button for each, and in-browser playback. Export
MIDI or send to the Deluge from the header.

The sound panel has one tab per area - **tune, lead sound, drums, bass,
chords, song, arrangement** - each with a sentence on what it shapes, the
steering words that apply to it as buttons, and its main controls. Every
slider names its two ends ("staccato ... legato"); hover a control for what it
does. The finer controls fold away under each tab. A tab marked ● has a
control moved in this take, and so does the control itself.

Transport: **space** plays and pauses (pause keeps the place), **←/→** jump
to the previous / next section, **Home** goes to the start, **Esc** stops.
Click anywhere on the ruler or the parts to jump to that bar; double-click a
section in the structure to play from it.

The readback is at note level: besides the parts that moved, it counts notes
*added* and *removed* per part, and the piano roll shades added notes and
ghosts removed ones, so a key change and a density nudge look different on
screen.

## Steering: twist, don't reroll

The sound panel is a steering surface, not a random button. Every part has a
*regenerate* button; the panel above it is for when the track has an identity
and you want to move it. The model, in `direct.py`:

- **Twisting a knob is a step, not a roll.** The engine keeps every seed
  frozen and recomposes, so the result is a small move from where you were -
  the bass got denser, the chords did not touch. A fresh roll is the opposite,
  and it is available, but it is a different action with a different name.
- **A roll starts a new take.** Regenerating a part or rerolling the song
  becomes the A/B baseline: from then on, the readback is measured against
  the roll, not the song you were steering into it.
- **You can walk the trail.** Every step is on a history; the page has undo
  and redo, and *reset take* puts every knob back where the take began.
- **The readback is the feedback.** After a change the page tells you exactly
  which knobs moved and which parts did - `moved drums, sub (drums.kick_density
  0.22->0.30, ...)`. Hold **A/B** to hear the start of the take and the current
  state in alternation.
- **Words work too.** An intent like `busier`, `down a third`, `faster by 4`,
  or a mode — `phrygian`, `dorian`, `phrygian dominant` — resolves to precise
  knob moves. A mode keeps the tonic and moves the whole harmonic personality
  (the b2, the raised 7th): a big move, on purpose. Anything the engine does
  not understand is said so rather than silently ignored.

## Profiles

`roots_dub` (72bpm one-drop), `steppers_dub` (78, four-to-the-floor),
`dubby_halftime` (88), `uk_garage` (132, 2-step), `dark_garage` (138,
proto-dubstep: 2-step kit under a dubstep arrangement), `dubstep_140`
(140 written / 70 felt). A corpus runs them in that order - sound system,
garage, dubstep: the music's family tree as a set.

A profile is one frozen dataclass in `profile.py` holding tempo, drum style,
register windows, harmonic rhythm and preferred bass figures. Genre-specific
constants live there and nowhere else, so adding a feel means adding a profile,
not editing the engine.

## Layout

```
theory/     pitch, scale, chord, voicing, harmony, rhythm, melody, bassline,
            fill, hook (forms, candidates, the hook score)
arrange/    structure — sections, kinds, energy curve, lengths
            interplay — the theme plan and who answers whom
            dynamics — rolls, risers, drop gaps, impacts, switch-ups
profile.py  genre settings
compose.py  the pipeline: key → progression → hook → plan → parts → Song
direct.py   steering — knobs, choice knobs, ops, word intents, mutation, trail
critique.py the self-critic: findings, fixes, the score, the revision loop
analyze.py  the listener: musical measures and one score to rank by
tells.py    where a producer would hear a machine: the giveaways, measured
listen.py   ears: any recording as numbers and spectrograms
corpus.py   a show: best-of-N songs, set ordering, variations, files
song.py     Note / Part / Song / Rig — the boundary between theory and MIDI
render/     standard MIDI file output, and an audio preview synth
deluge/     the AI SysEx protocol, client, and session-clip renderer
ui/         the steering page (standard library only)
```

## Deluge firmware

Depends on the custom `AISysex` command set, now in its own repo
(`deluge-firmware`, a small fork of the community firmware): a dedicated
`AI` SysEx namespace with 19 commands (note push/clear, session clip
select/clone/create/clamp, row expression and automation, preset load,
section repeat) plus the song-mode section loop that lets jam/song mode
wrap playback instead of stopping.

`deluge/sysex.py` is a direct transcription of the firmware's parser,
including the payload-length thresholds it uses to switch between 14-bit and
21-bit timing — `tests/test_deluge_sysex.py` pins all of them. The fork
removed the shared-sound param-manager hot path for performance; the
corresponding `SYNC_SHARED_OUTPUT_PARAMS` command is still in the protocol
but no-oped on the device, and riddim never sends it (cloned clips inherit
their template's output anyway).

## Tests

```bash
python -m pytest tests/ -q
```

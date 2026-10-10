# riddim

Theory-grounded dub and dubstep composition for the Synthstrom Deluge and a
modular rack.

This is a deterministic music-theory engine, not a model. Nothing is learned
from a dataset; every note comes from a rule you can read, argue with and
change. That is the point — when a bassline is wrong you can find the line
that made it wrong.

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
| `chords` | MIDI ch 1 → rack | Short offbeat skank stabs. Put this channel on the Nostalgia send. |
| `sub` | MIDI ch 2 → rack | Single oscillator. Forced monophonic — no two notes ever overlap. |
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
    --template-rows "chords=1,sub=2,lead=3,pad=4,drums=5,perc=6"
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
a build that never climbs, a hole that is loud.

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
`dubby_halftime` (88), `dubstep_140` (140 written / 70 felt).

A profile is one frozen dataclass in `profile.py` holding tempo, drum style,
register windows, harmonic rhythm and preferred bass figures. Genre-specific
constants live there and nowhere else, so adding a feel means adding a profile,
not editing the engine.

## Layout

```
theory/     pitch, scale, chord, voicing, harmony, rhythm, melody, bassline, fill
arrange/    structure — sections, energy curve, density and brightness
profile.py  genre settings
compose.py  the pipeline: key → progression → material → sections → Song
direct.py   steering — knobs, ops, word intents, the trail, the readback
critique.py the self-critic: findings, fixes, the score, the revision loop
song.py     Note / Part / Song / Rig — the boundary between theory and MIDI
render/     standard MIDI file output
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

# Roots dub against the judge: what moved, what didn't

October 2026, branch `library-kits`. The goal-panel question was why the judge
never hears our roots_dub as roots. This file records everything we tried. All
numbers come from the goal panel: roots_dub, 8 seeds (301–308), fresh compose,
sample library on. They are scored against the roots_dub references
(ml/judge.py, CLAP), and the text probes against the roots_dub + roots_reggae
references (ml/probe_gaps.py).

## Where it ended

| | FAD | closeness | real | heard as roots |
|---|---|---|---|---|
| before the drummer work (4aef7ac) | 0.361 | 0.808 | 0.84 | 0/8 (steppers 7, digital 1) |
| f21085c | 0.324 | 0.827 | 0.97 | 0/8 (steppers 6, digital 2) |

Roots now scores as very real dub, and FAD against its own references is the
best it has been. But the nearest family centroid is still steppers_dub or
digital_dub on every seed. roots_dub ranks third, 0.03–0.07 behind them.

## What moved the judge (kept)

| change | commit | FAD | real |
|---|---|---|---|
| a drummer: touches on 60% of riddim bars + 8.5 ms timing spread | df646bc | 0.361 → 0.342 | 0.84 → 0.92 |
| the kit lays back 3.3 ms (lean) | 52bc0cf / 6349062 | 0.342 → 0.341 | 0.92 |
| every other 4-bar block as rockers (kick on 1 and 3) | 1bf1ba1 | 0.341 → 0.336 | 0.92 → 0.91 |
| organ bubble −4 dB (BUBBLE_TRIM) | 314a667 | 0.336 → 0.332 | 0.91 → 0.95 |
| bass guitar's mid at half (BASS_MID) | 08b30d3 | 0.332 → 0.327 | 0.95 → 0.97 |
| sub a further −2 dB (TONE) | f21085c | 0.327 → 0.324 | 0.97 |

Earlier, for all profiles: the mix bus tone (11f009a), master drive to the
records' loudness (3b8b338, roots 2.4 dB) and dub drums −2 dB (e1c0b42).

## What didn't (reverted or left off)

- **Live vinyl one-shots** (rhythm-lab vinyl drumkit). With kick, snare, rim
  and hats from records: FAD 0.372, real 0.57. The loss comes from the
  snare/rim replacing the cross-stick; hats alone or kick alone are neutral.
  A softer kit bus and thinned hats did no better.
- **More touches** (drummer 0.9): FAD 0.343, real 0.88.
- **Hat-figure blocks** (drummer_blocks 0.5): FAD 0.348, real 0.88.
- **Keys:**
  - a drawbar organ multisample (Broken Beat Kit04): FAD 0.339, real 0.85.
  - the synth organ: FAD 0.354.
  - skank ±: noise to worse.
- **Space:** roots reverb ×0.6 lost on both counts.
- **Level:**
  - master drive 0: neutral.
  - a slow fader ride (RIDE 0.3–0.7): the 8-bar energy spread reached the
    records' 2 dB, but FAD was a hair worse and nothing else moved.
- **Bass guitar an octave lower:** FAD 0.336.
- **70s colour on the master:** top −3 dB, 40% width, wow, more tape.
  FAD 0.333–0.343. A mono kit did nothing.

## Measurements

The tools are ml/drum_feel.py, riddim.listen and ml/library_stats.py.

| | roots records | ours (latest) |
|---|---|---|
| 4-bar drum repeat (onset-grid Jaccard) | 0.54 | 0.69 (0.72 before the rockers blocks) |
| jitter_ms | 14.3 | 11.7 |
| lean_ms | −0.4 | −3.5 |
| 8-bar energy spread (p90−p10) | 2.0 dB | 4.7 dB |
| dynamic range | 5.1 dB | 7.0 dB |
| sub share (band) | 0.14 | 0.18 (0.43 before the tone EQ) |

For contrast, steppers records repeat at 0.82, and ours at 0.78.

## The readings that never moved

These are text-probe gaps against the roots records, in records' SD. They held
through every change above:

- "a drum machine loop" +1.8 to +2.0
- "a heavily compressed modern mix" +2.0
- "a piano" +1.7 to +1.9. It follows neither the organ, the skank guitar nor
  the bass mid.
- "organ stabs" +1.4, "a reggae organ bubble" +1.4
- "a big drop" +1.5, "a slow build" +1.6. These stay put even when the level is
  ridden flat, so they are not about dynamics.
- "a one drop reggae drum beat" +1.2 to +1.4 (ours more than the records)
- "a warm analogue 1970s recording" +1.2. Ours already reads warmer.
- "an MC toasting over the beat" −1.9. This is the one large negative.

## The likeliest reason: the voice

The references we are judged against are vocal music far more often than
steppers is. Share of records with the Demucs vocal stem above −15 dB against
the mix (references/stats/stems.jsonl):

| family | vocals median | records with vocals |
|---|---|---|
| roots_reggae | −8.8 dB | 70% |
| roots_dub | −13.6 dB | 58% |
| dubstep | −16.2 dB | 40% |
| steppers_dub | −21.5 dB | 38% |
| digital_dub | −19.4 dB | 34% |

An instrumental riddim with no MC and no singer sits nearer the mostly
instrumental steppers and digital-dub centroids, however roots its band
sounds. That is consistent with every band-side change improving FAD and
"real" without ever moving the rank. Voice is out of scope for now, so this
stays open. Two options until then:

- Judge roots against an instrumental-only subset of its references (vocal
  stem under −25 dB). That would tell us whether the band is right.
- Accept roots ranking third behind steppers as the ceiling for instrumental
  roots.

## Still open, beyond the voice

- "a piano" +1.8: no single part explains it. Candidates are the held chord
  voicings in the pad, or the organ's attack.
- "drum machine" +1.9 and "compressed" +2.0: unchanged by samples, timing,
  touches, repetition, tone or level.
- The drum repeat is at 0.69 against 0.54. Bigger figure changes (fills,
  dropouts, hats in and out) would be the next lever, if wanted.

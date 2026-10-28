# riddim against the plan's goals: checkpoint

28 October 2026, `library-kits` at b8d48a5. This checks the branch against the
goals in "riddim: putting the library to work".

**Start** is the first goal panel on this branch, at 7f56423 on 28 October,
before the mix-bus tone (4 songs a profile). **Now** is
`python ml/goal_check.py --out goals5 --per-profile 8`, run once plain and once
with `--instrumental` (8 songs a profile, seeds 301–308, sample library on). The
judge's FAD is pooled per family, and FAD grows with fewer songs, so the 4-song
start and the 8-song now are only roughly comparable.

## The goals

| goal (plan) | how we'll know (plan) | start | now | what's left |
|---|---|---|---|---|
| **Dub sounds played on keys** | dub "other" stem within 3 dB of the records | roots −8.3 dB, steppers −8.9 dB | roots −0.5 dB, steppers −1.3 dB | **Met.** The "roots" genre reading is still open: see the next row and docs/roots-notes.md. |
| **Garage reads as garage** | genre probe says UK garage before dubstep | uk_garage 0/4 heard as garage (dubstep, steppers, digital) | 0/8 against all references (dubstep 5, steppers 3); **8/8 against instrumental references** | **Met for the band.** The miss against all references is the voice: only 78 of 300 garage references are instrumental. |
| **A judge we trust more** | re-measured on the full library; its ranking agrees with Ed's picks on a blind A/B set | 439 reference tracks | Re-measured: 2,923 embedded references across 10 families, 2,530 records' stats, 462 Demucs stem sets, 229 sets. Known biases found and handled (below). | **The blind A/B against Ed's picks hasn't been done.** The one set of picks (the library A/B: Ed kept the library on 10 of 10) was not blind. The judge agreed on all three families' FAD and 8 of 10 pairs' closeness. |
| **Arrangements move like real sets** | section lengths, drop spacing and bass pulls match the radio and sound-system recordings | not measured | Sets measured (ml/set_stats.py): a change about every 3 minutes, a flat hour, 0.7–0.9 bass pulls per 32 bars. Live tracks now run 2.5–3.5 minutes and pull at that rate. 140 songs pull 0.6–1.4 per 32 bars (renders measure 0.63–0.66). | **Partly met.** Our songs swing between quiet and loud more than records do: the energy spread is 2.4× the records' in roots and 5× in uk_garage. Every flatter arrangement we tried lost FAD (the judge rewards the empty sections). |
| **Every family has a voice** | vocal stem within −13 to −24 dB, not −60 | −38 to −42 dB | −38 to −49 dB | **Open, out of scope on this branch.** It is the largest gap, and it explains most of the genre misses. |
| **The drums and voice are on the card** | a Deluge send plays the vocal rows from the riddim kits | – | not worked on here | Open. |

## The panel now (b8d48a5)

Stem levels, Demucs, as ours / records (gap):

| profile | drums | bass | other | vocals |
|---|---|---|---|---|
| roots_dub | −5.3 / −8.1 (+2.8) | −2.9 / −3.5 (+0.5) | −10.8 / −10.3 (−0.5) | −37.8 / −10.4 (−27.4) |
| steppers_dub | −4.2 / −7.0 (+2.9) | −2.9 / −2.4 (−0.5) | −12.9 / −11.6 (−1.3) | −39.2 / −20.2 (−19.0) |
| dubstep_140 | −1.2 / −6.5 (+5.3) | −10.1 / −3.8 (−6.3) | −14.3 / −12.6 (−1.7) | −48.8 / −16.2 (−32.6) |
| dark_garage | −1.6 / −6.3 (+4.7) | −7.9 / −5.0 (−3.0) | −12.8 / −13.5 (+0.8) | −47.0 / −15.3 (−31.7) |
| uk_garage | −2.8 / −4.8 (+1.9) | −5.0 / −4.9 (−0.0) | −14.9 / −13.0 (−1.9) | −44.8 / −12.2 (−32.6) |

Heard as its own family:

| profile | all references | instrumental references |
|---|---|---|
| roots_dub | 0/8 (steppers, digital) | 2/8 (as roots_reggae; own family 1st or 2nd on most) |
| steppers_dub | 8/8 | 8/8 |
| dubstep_140 | 6/8 | 6/8 |
| dark_garage | 3/8 | 0/8 (see below) |
| uk_garage | 0/8 | 8/8 |

dark_garage's instrumental column can't be read. dark_garage keeps 2
instrumental records and speed_garage 2, so both drop out of the instrumental
centroids, and only grime is left as dark_garage's own family.

The judge against each family's references (FAD / closeness / real):

| family | start, 4 songs a profile | now, 8 songs a profile |
|---|---|---|
| dub (roots + steppers, against steppers_dub) | 0.353 / 0.845 / 0.99 | 0.295 / 0.856 / 0.97 |
| dubstep (dubstep_140 + dark_garage) | 0.401 / 0.794 / 0.74 | 0.363 / 0.807 / 0.73 |
| garage (uk_garage) | 0.424 / 0.765 / 0.56 | 0.365 / 0.797 / 0.79 |

## Targets dropped

- **140 drums "+5 dB hot" in the stems.** On 28 October, dubstep drums −2 dB
  moved the Demucs drums stem by only 0.3–0.6 dB, and a shorter kick sub tail
  barely moved it. The bass stem rose as the drums stem fell. Demucs is filing
  our short, kick-locked 140 sub under drums, so this is an artifact of the
  measurement, not a mix level. Every thinner kick also cost "real" (0.73 →
  0.61–0.70). Leave the 140 drums alone, and don't use the 140 drums/bass stem
  gap as a target.
- **Roots "era" colour** (duller top, narrower, wow, more tape) lost. Ours
  already reads warmer than the records. See docs/roots-notes.md.

## What we learned about the judge

These are the judge's known biases, all measured on this branch:

- **Level.** Raising dubstep and garage renders by 3 dB moved "real" from 0.54
  to 0.76 and 0.41 to 0.60. Every profile is now driven to its records'
  loudness (render DRIVE).
- **Voice.** Vocal-heavy reference families (roots_reggae 70% with a vocal,
  roots_dub 58%, uk_garage) pull instrumental riddims toward the instrumental
  families. Report genre both ways: `goal_check --instrumental`.
- **Variety over typicality.** Filling uk_garage's empty sections raised
  closeness (0.795 → 0.830) but worsened FAD (0.357 → 0.382–0.395), because FAD
  compares the spread of 10 s windows. A flatter fader ride did the same for
  roots.
- **The text probes are not about dynamics.** "A big drop" and "a slow build"
  didn't move when the level was ridden flat. "A drum machine loop" and "a
  heavily compressed modern mix" didn't move with any drum or tone change.
- **Noise.** FAD moves under about 0.005 on 8 seeds are noise. Calls of that
  size were left to the other measures.

The plan's gate still applies: the judge only keeps or rejects changes once it
has matched Ed's blind picks, and that check is still to do.

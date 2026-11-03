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
| **A judge we trust more** | re-measured on the full library; its ranking agrees with Ed's picks on a blind A/B set | 439 reference tracks | Re-measured: 2,923 embedded references across 10 families, 2,530 records' stats, 462 Demucs stem sets, 229 sets. Known biases found and handled (below). | **Not passed.** On the blind A/B (12 of its calls, below), Ed agreed with the judge's calls on 5 of 9 pairs he could tell apart (sign test p = 0.50), and couldn't tell 3. The judge's own score on each clip agreed with him on 6 of 9. The judge stays a guide, not a gate. |
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
has matched Ed's blind picks. That check has now been run, and it has not
matched (below).

## The blind A/B (28 October, ml/blind_ab.py)

This covered 12 of the judge's keep-or-revert calls. Each pair is the same
seed made both ways, the same ~38 s from two bars before the first drop,
matched in loudness, with sides shuffled. "Ed picked" says which side he
chose: kept is the judge's side, other the side it turned down. The last
column is the judge's closeness score on the clip itself, kept / other.

| # | profile | change | call | Ed picked | judge on the clip (kept / other) |
|---|---|---|---|---|---|
| 0 | roots_dub | a drummer: touches and a player's timing | clear | kept | 0.798 / 0.780 |
| 1 | roots_dub | every other four bars as rockers | close | **other** | 0.860 / 0.859 |
| 2 | roots_dub | organ bubble 4 dB down | close | kept | 0.844 / 0.841 |
| 3 | roots_dub | bass guitar's mid at half | close | kept | 0.855 / 0.852 |
| 4 | steppers_dub | dub drums 2 dB down | close | **other** | 0.861 / 0.863 |
| 5 | uk_garage | reverb and delay ×2.5 | clear | kept | 0.776 / 0.790 |
| 6 | uk_garage | short chord stabs, chords 4.4 dB down | close | kept | 0.784 / 0.785 |
| 7 | uk_garage | Rhodes instead of organ | split | **other** | 0.821 / 0.823 |
| 8 | uk_garage | master drive to the records' loudness | clear | **other** | 0.8415 / 0.8421 |
| 9 | dubstep_140 | master drive to the records' loudness | clear | can't tell | 0.809 / 0.803 |
| 10 | dark_garage | electric piano instead of the dub organ | clear | can't tell | 0.777 / 0.790 |
| 11 | dubstep_140 | 140 drums 2 dB down (rejected) | close | can't tell | 0.755 / 0.748 |

- **The judge's calls:** Ed agreed on 5 of 9 decided pairs (56%). A coin
  would do as well (one-sided sign test p = 0.50).
- **The judge on the clips:** it agreed with Ed on 6 of 9.
- **Kept changes Ed preferred undone:**
  - the roots rockers blocks (1). The clip judge sided with the call, by
    0.001.
  - the dub drums −2 dB (4). The clip judge sided with Ed.
  - the uk_garage Rhodes (7). The clip judge sided with Ed. This is the one
    call made against the usual FAD rule, on a large "real" gain.
  - the uk_garage master drive (8). The clip judge sided with Ed, by 0.0006.
    With loudness matched, this pair hears only the drive's saturation, not
    the level the judge rewarded.
- **Can't tell:** the dubstep drive, dark garage's electric piano and the
  rejected 140 drums trim. Loudness-matched, those changes were inaudible to
  him.

So at this resolution the judge is not a proxy for Ed's ears. Its calls
mostly sit inside differences he either can't hear or hears the other way.
What follows:

- Keep the judge for coarse questions (genre ranking, large FAD moves, level),
  and stop letting it decide close calls.
- Reconsider the four reverted-by-Ed changes on listening, not on the
  judge's numbers.
- A bigger blind set, with more of the clear-win kind, would say whether the
  judge is right on the big calls at least: 12 pairs can only show a strong
  agreement, not a weak one.

## Checkpoint at f40ffac (30 October): the gaps that are left

`goal_check --instrumental --seeds 16` (renders in goals6/), then
`ml/gaps.py goals6`: every measured target's mean over the 16 songs against
the records', in units of its tolerance (ml/targets.py).

| profile | own family | FAD | closeness | real |
|---|---|---|---|---|
| roots_dub | 8/16 | 0.326 | 0.798 | 0.98 |
| steppers_dub | 12/16 | 0.327 | 0.844 | 0.99 |
| dubstep_140 | 2/16 | 0.376 | 0.790 | 0.87 |
| dark_garage | 0/16 | 0.425 | 0.764 | 0.55 |
| uk_garage | 8/16 | 0.387 | 0.768 | 0.61 |

The largest gaps, ranked. Ours, then the records' figure in brackets:

| rank | profile | target | ours (records) | tolerances | kind |
|---|---|---|---|---|---|
| 1 | dubstep_140 | bass stem under 60 Hz | 0.547 (0.898) | −35 | artifact: Demucs files our 140 sub under drums |
| 2 | dark_garage | "other" 500–2500 / 150–500 Hz | 0.786 / 0.150 (0.448 / 0.487) | ±34 | artifact: our clean "other" bus measures 0.50 / 0.47 (dark garage seed 2); Demucs on the full mix files its low mids elsewhere |
| 3 | dubstep_140 | "other" 500–2500 Hz | 0.762 (0.459) | +30 | probably the same artifact |
| 4 | roots_dub | drum stem snare / kick share | 0.532 / 0.360 (0.276 / 0.567) | +26 / −21 | **real: next to work on** |
| 5 | roots_dub | bass onset-strength spread | 1.025 (0.611) | +21 | real |
| 6 | dubstep_140 | bass stem 60–120 Hz | 0.282 (0.057) | +23 | artifact, the other side of 1 and 3 |
| 7 | uk_garage | mix sub share | 0.345 (0.246) | +20 | real |
| 8 | uk_garage | "other" 500–2500 Hz | 0.688 (0.502) | +19 | real: the organ (brightening it not kept) |
| 9 | dark_garage | "other" held | 0.552 (0.902) | −18 | artifact: 0.99 on the clean "other" bus; an organ held under the stabs moved nothing |
| 10 | steppers_dub | bass stem 60–120 Hz / under 60 | 0.672 / 0.194 (0.507 / 0.346) | +16 / −15 | real |
| 11 | dubstep_140 | snare body | +3.7 (−12.2) dB | +16 | real |
| 12 | steppers_dub | drum stem snare share | 0.407 (0.269) | +14 | real |

Blocked on voice: the vocals stem sits 22–31 dB under the records' in every
profile. In probe_gaps, "chopped and pitched vocal samples" is the widest
word gap in dubstep and dark garage. probe_gaps' other words are loose: "a
big drop" and "a drum machine loop" are over in most profiles, and "a piano"
is over in roots (+2.3 sd).

## Re-baseline with the audible lead (1 November, 8 songs a profile, seeds 301-308)

The louder lead moves the full mix's band shares, and Demucs files our synth
lead under "other" where the records' leads and vocals go to "vocals" - so
from now on the stem targets are measured on the same song rendered with its
lead muted (ml/targets.py write_nolead; ml/gaps.py and targets.ab do it),
and the full-mix targets with it. Muted, the "other" stem is back where it
was on 30 October (dubstep/dark garage 500-2500 Hz 0.74/0.75, then 0.76/
0.79): those gaps predate the lead. The largest gaps now:

| profile | target | ours | records | gap (tolerances) | note |
|---|---|---|---|---|---|
| dark_garage | other_lowmid | 0.153 | 0.487 | -33.5 |  |
| dubstep_140 | bass_sub | 0.591 | 0.898 | -30.7 | artifact |
| dark_garage | other_mid | 0.746 | 0.448 | +29.8 |  |
| dubstep_140 | other_mid | 0.737 | 0.459 | +27.7 |  |
| dubstep_140 | bass_low | 0.278 | 0.057 | +22.1 |  |
| roots_dub | snare_share | 0.496 | 0.276 | +22.0 |  |
| uk_garage | other_mid | 0.705 | 0.502 | +20.3 |  |
| dark_garage | other_held | 0.510 | 0.902 | -19.6 |  |
| uk_garage | bass | 0.394 | 0.487 | -18.5 |  |
| steppers_dub | bass_low | 0.685 | 0.507 | +17.8 |  |
| uk_garage | bass_sub | 0.241 | 0.413 | -17.2 |  |
| dubstep_140 | snare_body_db | 4.286 | -12.156 | +16.4 |  |
| roots_dub | bass_cv | 0.933 | 0.611 | +16.1 | advisory |
| steppers_dub | high_mid | 0.130 | 0.050 | +16.0 |  |
| roots_dub | cv_kick | 0.829 | 0.516 | +15.7 | advisory |
| dubstep_140 | other_lowmid | 0.164 | 0.321 | -15.7 |  |
| uk_garage | bass_cv | 1.044 | 0.747 | +14.9 | advisory |
| steppers_dub | other_high | 0.207 | 0.060 | +14.7 |  |
| roots_dub | high_mid | 0.169 | 0.097 | +14.4 |  |
| steppers_dub | other_held | 0.484 | 0.770 | -14.3 |  |


Kept changes to recheck under the audible lead (they were judged on full-mix
band shares when the lead was 24-30 dB down): roots' master mids refit and
kit trims (b04e813); the dub snare and sub trims (eda2fb9). The skank and
returns low-passes (8f26d52, 362c2b0, fd175b6) were judged on the "other"
stem, which the lead-muted split keeps comparable.

Rechecked (seeds 101-116, the change undone against head): both stay.
Undoing b04e813 put the high mids, the hats' brightness and the "other"
stem away; undoing eda2fb9 put the kick and snare shares, the sub, the bass
band and the loudness away. Roots' low mids now sit a little over the
records' (0.191, records 0.163) with the lead in them.

## The bass as the hook (3 November, on for dubstep_140: profile bass_hook)

From an EDMProd 140 tutorial (notes in the thread): in this style the bass
line is the hook. `bass_hook=True` on a profile: in the peaks the sub plays
the hook's motive in eight-bar phrases that breathe (a gap at the phrase
start, the second four varied, a drop-out at the end), the lead teases in
the intro and answers in the drop-outs (the model writes no lead), the growl
holds through a phrase and moves at its end with a distorted mid-bass layer,
and the drops differ (the second opens without kick and clap for 4 bars,
the wobble thins later on). Blind against dubstep as it shipped (the
fine-tuned model's lead), ml/audition_ab.py basshook: Ed picked it 4-2
(p = 0.69 by chance: a lean), so it is on for dubstep_140. `bass_hook=False`
in riddim/profile.py undoes it.

Ed discarded what he hadn't listened to (3 November): dubstep's kick body
and roots' held organ 1.6 are reverted (the knobs stay), and the step-3000
hook test is gone. The lead's 3.5 kHz low-pass stays: it corrected the
overshoot from his own "too quiet" call.

## Ed's calls (over the targets)

- **2 November, the audition pages** (ml/audition_ab.py, ml/hook_ab.py):
  - Roots' louder, shorter kick (dc59172): Ed picked it 2-0 - stays.
  - uk's deeper sub (candidate): the old one once, can't tell once - not taken.
  - The darker lead (8e2744b): no picks - stays as measured.
  - Dubstep's snare: Ed picked a thin library snare over the synth's in 6
    of 8 pairs; Tape Treated's Cassette Snare 05 and Nu Garage's Snare11
    2-0 each. Dubstep now plays the cassette snare (its body nearest the
    records'); dark garage keeps its own, unheard.
  - Garage hooks, the garage fine-tune (models/g1) against the base model:
    4-5 - the base stays the garage hook writer.
  - Steppers' bass: Ed picked UK Urban's Bassaw Multi, the SH101 Sub and
    Nu Garage's Org_Bass 2-0 each over the synth sub (Birdy Sub lost 0-2).
    Steppers now plays the Bassaw (nearest the records' bass stem); roots
    keeps its synth sub, unheard.
  - Not yet picked: the step-3000 hooks (:8778).

- **The trained model writes the drops' hooks by default (1 November),
  routed by profile** from Ed's two hook blind tests: the dub fine-tune
  (models/v3/best) for roots, steppers, dubstep and dubby half-time, which
  won the dub and dubstep pairs 3-0 once tuned; the base music-medium-800k
  for dark and uk garage, which won garage 3-1 before tuning and lost it
  0-3 after. Pooled, the model 9, riddim 6, 3 can't tell (p = 0.61): a
  lean, not a result, so it is one flag to undo - `--rule-lead` or
  RIDDIM_MODEL_LEAD=off - and riddim's own lead comes back by itself when
  ~/ml or the model is missing. The CLI only; the UI keeps its "model
  lead" button (a model take per press costs ~15 s, too slow for every
  twist).

- **The drops' lead up 18-23 dB (1 November, `LEAD_LIFT_DB`).** Ed, on the
  hook blind test: "the leads are too low in the mix, I can't even hear
  them". Measured on seed 401's drops, while the lead plays: it sat 24-30 dB
  under the full mix, more than 15 dB under it 97-100% of the time. Now
  about 6 dB under it (median, every profile), more than 15 dB under 11-15%
  of the time, with a 2-4 kHz presence lift and the stabs and pad ducking
  up to 30% under it. Exempt from the targets rule: it moves the band and
  stem shares, and the targets don't get a vote.
  - **Then darkened (same day):** the lifted lead had pushed the mix's 1-5 kHz
    share far past the records' (uk 0.041 -> 0.131, records 0.069; roots
    0.278, records 0.097). Demucs files our synth lead under "other", not
    "vocals", so the records' "other" and "vocals" together are the
    comparison: on that, the lead sat 0-1 dB hot in dubstep, dark garage
    and roots and ~3 dB hot in steppers and uk. Now low-passed at 3.5 kHz
    with no presence lift (centroid ~2.6 -> ~2.0 kHz), the dub leads 2 dB
    down: 8-9 dB under the mix while it plays. Needs Ed's ears: if it's
    buried again, the level comes back before the brightness does.

## The hook blind test (1 November, ml/hook_ab.py)

Ten pairs, each one song with riddim's lead and with the base model's
(music-medium-800k, not fine-tuned) in the drops, the leads level-matched.
Ed preferred the model's hook in 6, riddim's in 3, and couldn't tell in 1
(p = 0.51 by chance: not yet a result). By style: the 140 and garage
profiles went to the model 5 of 6 (dubstep 2/2, uk 2/2, dark garage 1/2);
the dub profiles 1 model, 2 riddim, 1 can't tell. The fine-tune (models/v3)
gets the same test when it finishes.

The fine-tune (models/v3, best at step 6,000: Lakh's matched set filtered
for the DMZ sound plus riddim's band parts; stopped at 9,000 when its lead
loss turned up) on fresh seeds 405-406: Ed picked 8 of 10 pairs - model 3,
riddim 3, can't tell 2. By style it reversed the base model's pattern: the
model won roots, steppers and dubstep (1 each; dubstep's other can't
tell), riddim won dark garage 1 (1 can't tell) and uk 2. The data it was
tuned on is dub and half-time 140, so better there and worse in garage is
what you'd expect; with this few pairs neither pattern is a result yet.

## Kept on measured targets (29-30 October), to audition

After the judges failed the blind test, changes were kept only when they
moved a measured target toward the records. The method:

- ml/targets.py, 16 seeds paired
- a 95% interval on each target, with a tolerance for collateral moves
- drums, bass and "other" measured on Demucs stems split the same way as
  the records' (ml/drum_stem.py)

Each entry gives the commit, what changed, and the effect as before → after
(records).

- **c6ee02f: a drummer's timing for steppers, dubstep, dark and uk garage.**
  Each kit hit is a few ms off the grid and laid back.
  - jitter: steppers 10.8 → 11.3 ms (13.0), dubstep 9.5 → 11.6 (13.9), dark
    garage 8.8 → 11.1 (13.6), uk garage 11.9 → 12.5 (13.7)
  - lean: steppers −6.9 → −5.2 ms (−4.1), dubstep −8.6 → −6.1 (−1.8), dark
    garage −9.6 → −7.1 (−2.6), uk garage −6.2 → −4.4 (−2.6)
- **339c461: steppers' snare body ×0.3 and hats ×1.35.**
  - snare body against its crack: +8.3 → +6.7 dB (+0.4)
  - hat share of the drum stem: 0.058 → 0.064 (0.082)
- **4bc7c07: uk garage's snare high-passed at 350 Hz, body ×0.3, tone sub
  −1.5 dB.**
  - snare body: +6.0 → +3.9 dB (−2.8)
  - snare share: 0.43 → 0.31 (0.28)
  - kick share: 0.50 → 0.63 (0.58)
- **6f0f982: roots and steppers bass notes 0.8 as long.**
  - how often the sub sounds: roots 0.751 → 0.737 (0.694), steppers 0.780 →
    0.768 (0.691)
- **b8ea49a: steppers' organ bubble +4 dB.**
  - "other" stem 2.5–10 kHz share: 0.191 → 0.174 (0.060)
  - its 150–500 Hz share: 0.316 → 0.344 (0.356)
  - how often it sounds: 0.427 → 0.440 (0.770)
- **87b96bf, 59fbbd0: chords' reverb and delay sends ×2 in roots, steppers,
  dubstep and dark garage.**
  - how often the "other" stem sounds: roots 0.394 → 0.436 (0.768), steppers
    0.438 → 0.473 (0.770), dubstep 0.436 → 0.519 (0.747), dark garage 0.438
    → 0.554 (0.902)
  - These are the most likely to read as "more wash" by ear. The judge
    disliked more reverb before (dubstep space ×1.6, dark garage ×2.5).
- **Listen to this one first, the biggest audible change today: the
  held organ, roots and steppers.** An organ holds each chord under the stabs, from its first
  stab to the next chord (audio.HELD_ORGAN). It's a new sound in the
  song, not a level change, so it's the one most likely to be wrong to the
  ear.
  - how often the "other" stem sounds: 0.548 → 0.708 (0.768); steppers, at
    half the level, 0.560 → 0.599 (0.770)
  - its 150–500 Hz share 0.237 → 0.278 (0.384), 500–2500 Hz 0.553 → 0.529
    (0.476), 2.5–10 kHz 0.210 → 0.193 (0.087)
  - The mix's drum lean moved away, −2.2 → −3.2 ms. The drums-only bus's
    lean didn't move (seeds 1–5): the organ's sustain shifts the onset
    picking, not the drums. Lean and jitter are advisory for changes that
    leave the drums alone, once a drums-only-bus check shows them unchanged
    (ml/targets.py).
- **Roots' skank darker and fuller** (5 kHz low-pass, −3 dB at 2.2 kHz, +3 dB
  at 500 Hz), and darker plate and spring returns in roots and steppers
  (3.5 kHz). A tone change you'll hear on every offbeat.
  - roots' "other" stem 2.5–10 kHz share 0.189 → 0.114 (0.087), 150–500 Hz
    0.281 → 0.385 (0.384), centroid 2275 → 2018 Hz (1884)
- **Roots' kick 1.6x louder with its sub tail cut short (160 ms), hats
  x1.7, drive 3.6** (1 November, seeds 101-116): the drum stem's kick share
  0.408 -> 0.478 (records 0.567), snare 0.508 -> 0.444 (0.276), the mix's sub
  0.140 -> 0.109 (0.102), bass 0.494 -> 0.539 (0.548). You'll hear a
  punchier, tighter kick. Kept over one move away - the hats' hit spread in
  the mix (0.548 -> 0.524) - that the clean drum bus shows is the detector,
  not the hats (0.538 -> 0.536).
  - Against it: on the clean bus the drums' share of the mix's 60-150 Hz
    went 39% -> 58% (8 seeds); the records' is 24% (from their stems).
    Inside the drums the kick now sits nearer the records', but the drums
    as a whole are louder in the mix than the records' (-5.1 against -8.1
    dB). If it sounds too kick-heavy, that is why - the drums down, rather
    than this undone, is the next thing to try. (Not the drums' level: Ed
    kept the dub drums at full level. Nor the kick's body: at -2 dB, seeds
    101-116, the mix's 60-250 went under the records' (0.539 -> 0.531,
    records 0.548) and the sub and kick share away - reverted; roots' mix
    bands are already on the records' with the kick as it is.)
- **Roots' master mids refitted (low mids +1.1 dB, high mids −0.9 dB), with
  the snare ×0.55 and hats ×1.15** (ml/fit_tone.py on seeds 1–8, confirmed
  on 101–116). Low mids 0.141 → 0.159 (0.163), high mids 0.111 → 0.096
  (0.097); the "other" stem's band shares all toward.
- **eda2fb9: roots' snare ×0.6 in the kit, sub 1.5 dB lower.** Drum stem
  kick share 0.346 → 0.388 (0.567), snare 0.541 → 0.515 (0.276).
- **44a5ebc, 85eee4c: the organ bubble through every section the chords
  play in, roots and steppers; steppers' bubble back at its plain level (the
  +4 dB of b8ea49a removed).**
  - how often the "other" stem sounds: roots 0.534 → 0.550 (0.768), steppers
    0.520 → 0.558 (0.770)
  - steppers' "other" 150–500 Hz share: 0.351 → 0.346 (0.356); with the
    +4 dB kept it overshot to 0.399

Tried and not kept:

- uk's sub an octave's worth deeper (Register 24-45, as dubstep's; it sat
  at a median 73 Hz, its fundamental in the 60-150 band), with the kick's
  sub cut short (250 ms), the sub x1.4, the kick x1.2, drive 5.3 and the
  master's mids 0.6 dB down - six rounds on seeds 101-116. The bass stem
  lands on the records' (sub share 0.232 -> 0.425, records 0.413; 60-150
  0.465 -> 0.430, 0.415; 150-500 0.240 -> 0.112, 0.110) and the mix's mids
  toward, but the sub band held longer (0.789 -> 0.815, records 0.669) and
  the bass stem's notes shorter: can't tell. The closest candidate yet for
  uk's bottom end - one to hear rather than measure further.
- The same deeper sub in steppers (Register 24-45): reverted. Its bass stem
  lands on the records' too (sub share 0.170 -> 0.382, records 0.346; 60-150
  0.642 -> 0.494, 0.507), but the mix's sub band 0.270 -> 0.339 (0.264), its
  60-250 band 0.471 -> 0.378 (0.500) and the sub held longer. In both, the
  records keep a deep bass stem and a full 60-250 band at once: the 60-250
  energy our mixes lack when the sub goes deep must come from somewhere else
  in the records' band (the kick's body, the drums) - the next thing to
  measure, not the sub. With the kick's sub cut short as well (300 ms):
  reverted - the mix's sub held, but its 60-250 went further under (0.471 ->
  0.417, records 0.500), its mids and the sub's hold away. A deeper sub in
  steppers needs the 60-250 the records' bass guitar supplies. With the
  bass guitar x1.5 too: reverted, the 60-250 barely moved (0.417 -> 0.420)
  - our bass guitar is too quiet a part to carry it. Steppers' bottom end
  wants a fuller bass voice (a sampled bass guitar, say) more than levels.
  - Measured (1 November, from each stem's level against the mix and its
    60-150 Hz share; ours from the lead-muted split): the share of the
    mix's 60-150 Hz that the drums supply is ours 34% / records 24% in
    roots, 49% / 30% in steppers, 93% / 81% in dubstep, 91% / 57% in dark
    garage, 64% / 59% in uk. The records' 60-150 comes more from the bass,
    and our dubstep and dark garage carry 3-5 dB more of it than the
    records', nearly all kick: the kick wants thinning there, and a deeper
    sub then wants the bass's 60-150 back.
- Dubstep's snare high-passed (900 Hz; 700 Hz with the snare x1.4), hats
  x1.3, seeds 101-116: the snare's body +3.7 -> -4.9 dB (records -12.2), low
  mids and hats toward - but its 150-2500 Hz share past the records' (0.328
  -> 0.236/0.258, records 0.306) and the "other" stem's held share down
  (0.48 -> 0.43, records 0.75), the cut taking out what Demucs had filed as
  "other". The records' snares are thin and still fill 150-2500: a brighter,
  crackier snare sample, not a filter, is the likely fix.

- uk's kick sub decay (229 ms) fitted with the master tone (ml/fit_tone.py
  --decay, seeds 1-8: loss 194 -> 56): reverted on 101-116, high mids
  0.155 -> 0.258 (records 0.069). The fit ran across the lead lift (Ed's
  call, c3b65cf), which by itself took uk's high-mid share 0.041 -> 0.155:
  the gap table needs re-baselining with the louder lead before any more
  band-share tuning.

- The master tone fitted jointly (ml/fit_tone.py, seeds 1–8, every band
  within ±3 dB, confirmed on 101–116). Steppers and uk didn't pass, and in
  roots only the mids passed (kept above). Any sub cut on the master takes
  the kick's 60–150 Hz with it. uk's fit brought its sub to the records'
  (0.348 → 0.249, records 0.246), its top end near them (high mids 0.041 →
  0.082, records 0.069; air 0.010 → 0.021, 0.023) and its hats too, but the
  bass band, the loudness (−1.8 dB) and the organ's body moved away, and it
  needed +10.7 dB at 2.2 kHz and +9.2 dB of air on the master. uk's dull top
  end is the sources' (the organ, the hats), not something master EQ should
  fix.

- louder hats elsewhere
- the snare change in dubstep
- bass registers lowered (the bass stem moved toward the records, but the
  mix's sub moved away)
- shorter bass in the 140 profiles (no measurable move)
- every reverb return ×1.6
- a bigger bubble
- chords low-passed at 3 kHz in roots (brightness right, the mids away)
- longer skank notes (no measurable move)
- uk garage's dull "other" stem: the garage chords trim off; the organ
  voiced a fifth or an octave up; its top note doubled an octave up, with or
  without a 500 Hz lift. Each brightened it toward the records but thinned
  its 150–500 Hz share away from them.

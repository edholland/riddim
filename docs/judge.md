# The judge, rebuilt (4 November 2026)

Ed's brief: the judge should work, with no doubts, in a tight loop that
doesn't need him. It also has to work on real recordings of the rig.

**What changed.** The judge is now `riddim/judge.py`. It is plain DSP with
no torch and no downloads, and it runs on any recording, including L6max
multitrack stems. It answers **A**, **B** or **can't tell**. It only names a
winner when two separately fitted judges agree, and when the gap is past a
line set on held-out data.

**How it scores.** It was tested on 954 real records by 754 artists. In
each test pair, one side was the record as released. The other was the
same record with one known flaw added (24 flaw types, three sizes each). On
artists it had never seen:

- When it named a winner, it picked the record **97.6% of the time** (95%
  interval 97.0 to 98.1%).
- It never named a winner on a pair that differed only by a 0.5 dB nudge
  (0 of 1,908).
- The old approach ("nearer the records' centre wins") picked the record
  **57% of the time**. That is close to a coin flip, and it explains why
  the old judge matched Ed only 5 times out of 9.

All the numbers are in [judge-tests.md](judge-tests.md), and
`ml/judge_fit.py` regenerates them.

## Why the old judge failed

Three faults showed up in the evidence:

1. **It made calls inside its own noise.** The blind A/B calls it got wrong
   turned on closeness gaps of 0.0006 to 0.003. Its own seed-to-seed noise
   was about 0.03, ten times larger. Nothing told it when to say "can't
   tell".
2. **"Closer to the records' centre" measures typicality, not quality.** On
   the new test set it picks a record over the same record with a flaw added
   only 57% of the time. A flaw often moves a record *toward* the average:
   added hiss makes a clean record more like the hissy dub records, and an
   EQ boost can pull an unusual master toward the centre. The docs already
   showed the same symptom: filling sections raised closeness and made FAD
   worse.
3. **No one checked it against known answers before trusting it.** CLAP is
   a general audio-text model and is not sensitive to mix level ("matching
   stem balance alone does not move CLAP"). The later real-vs-riddim critic
   reached a held-out AUC of 1.000 because it learned the renderer's
   fingerprint. It put every riddim song at about −12 log-odds, so its
   ranking *between* two riddim takes was arbitrary. It scored 11 of 19
   against Ed.

## How the new one works

- **Ears** (`riddim/ears.py`) take 80 named measurements on each 8-second
  window, after setting loudness to −14 dBFS RMS. They cover:
  - tonal balance in 18 half-octave bands;
  - how much each band moves;
  - punch, tails, stereo width and grit in sub, bass, low mid, high mid and
    air;
  - where each band's envelope modulates (swells, beats, sixteenths, buzz);
  - crest, loudness range, brightness and attack density.

  `ears.describe()` reports these in words. `ears.diff()` says what changed
  between two takes, in dB.
- **Known flaws** (`riddim/defects.py`) come in six groups:
  - EQ: sub, bass, mud, honk, harsh, scoop, air, dull;
  - bandwidth: low-pass, high-pass;
  - dynamics: drive, clip, squash, smear, pump;
  - space: reverb, dub echo;
  - stereo: mono, too wide, sub pushed into the sides;
  - noise: hiss, hum.

  Each flaw has three sizes.
- **Two judges** (`ml/judge_fit.py`), each fitted so the record beats its
  flawed copy:
  - `dist` scores only the distance from the style's records, measured in
    the records' own spread, with a soft cap so one wild measurement can't
    outvote the rest. No direction counts as better than another.
  - `full` also learns directions, such as "less grit in the bass" or
    "drier tails".

  Alone, each is about 94% right when it calls. On its own, `full` would
  prefer less drive every time, which would be wrong for the Malevolent.
  Making the two agree removes that bias and lifts precision to 97.6%.
- **Calling line.** Each judge's minimum gap is set on artists held out from
  its own fit. The panel needs a gap of at least 0.87 for `dist` and 0.48 for
  `full`, and a paired t of at least 3 across windows. A pair where only one
  judge calls comes back as "lean", which is not a winner.

## What it can't do (read this before trusting a call)

- **Flaw types it never saw.** Each group was held out of the fit in turn
  and then tested. The panel stayed reliable on held-out bandwidth (96%),
  space (94.5%), stereo (100%, from 24 calls) and EQ (90.5%). It failed on
  held-out dynamics (21% of 98 calls) and noise (3% of 30). So a call is
  only as good as the catalogue's coverage of the *kind* of change. Every
  rig knob the loop turns needs a flaw of that kind in `defects.py`, then a
  refit. Filter cutoff, drive, delay, level, EQ and width are covered now.
  Wavefolding and the Fusion's modulation are not yet.
- **Coverage is low on purpose.** It names a winner on 14% of flaw pairs.
  The obvious ones get called: drive (63 to 70%), low-pass (59 to 80%),
  high-pass at 70 to 120 Hz, mono collapse. A 2 dB EQ move never does, and
  even an 8 dB sub or bass move only gets called 12 to 19% of the time. The
  records themselves spread that wide, so an absolute judge can't know
  which level is right. For level questions, use the stem balance below.
- **Notes, arrangement, hooks and instrument choice.** These are outside
  what it hears. On Ed's 22 blind pairs it named a winner once. That was the
  140 drums 2 dB trim, a pair Ed couldn't tell apart. It said "can't tell"
  or "lean" on the other 21, and never contradicted a pair Ed decided.
  - On the sample library against the synths (Ed chose the library 10 out of
    10), both judges lean slightly toward the *synths*, below the calling
    line.
  - Whether a part sounds like a real instrument is not a mix property this
    judge measures. That stays with Ed's ears, or with a different tool.
- **References are records only.** The references are 2 × 20-second
  excerpts per record, Opus-coded at 160k, from about 120 records per family
  (12 for dark garage). Dark garage's centre is mostly the pooled one.

## On the rig

```
riddim-cli judge takeA.wav takeB.wav --style dubstep_140        # which is better, and what changed
riddim-cli judge take.wav --style roots_dub                     # one take: where it's off the records
riddim-cli judge --stem drums=ch1.wav --stem bass=ch3.wav \
    --stem chords=ch5.wav --stem lead=ch7.wav [master.wav] --style steppers_dub
```

- **Stems.** `--stem` reads an L6max multitrack take. Roles map to the
  records' Demucs stems: drums, bass (with `malevolent` and `sub` as
  aliases), other (chords, pad, lead, coral, delay, fx) and vocals. The
  report covers:
  - the mix against the style's records;
  - each stem group's level against the mix, set beside the records'
    middle half (`riddim/data/stem_levels.json`, 50 records a family);
  - each channel in words, with its raw peak and RMS, so clipping on the
    desk is visible.

  The rig's stems are clean and the records' stems are Demucs splits, so
  the quieter stems read a little low beside them.
- **Aligned takes.** For A/B, the two takes should be the same bars, as two
  takes of one Deluge pattern are. The comparison is then paired window by
  window, which is what makes small differences callable. Use
  `--unaligned` otherwise. It is less sensitive.
- **The loop should act only on a winner.** Treat "lean" and "can't tell"
  as no change. In Python, `judge.compare(a, b, style)` returns `.winner`
  ("A", "B" or None), `.confidence`, `.reasons` and `.text`.

## Reproduce

On the Mac, where `references/` lives:

```
python ml/judge_excerpts.py --per-family 120             # 2 x 20 s excerpts a record -> references/judge2/
python ml/judge_data.py refs --meta references/judge2/refs.json --audio references/judge2/refs \
    --out references/judge2/judge_feats.npz --flaws 12   # ~45 min on 4 cores
python ml/judge_ed.py --blind ab-blind --library ab --out ed_feats.json
python ml/judge_fit.py --feats references/judge2/judge_feats.npz --ed ed_feats.json --write
                                                         # -> riddim/data/judge.json, docs/judge-tests.md
```

The measured features this model was fitted on (`judge_feats.npz`, 31 MB,
numbers only) are in the project's shared files (`judge/judge_feats.npz`); Ed's pair features are in `references/judge2/ed_feats.json` on the Mac. The fitted model is
74 KB of JSON.

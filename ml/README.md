# ml - the model behind the model hooks

Everything here runs in its own environment (`~/ml/venv`: torch with MPS,
transformers, mlx, mido, scipy, and the `anticipation` package from
https://github.com/jthickstun/anticipation installed with `--no-deps`). Data,
checkpoints and outputs live under `ML_HOME` (default `~/ml`), never in the
repo.

| file | what it does |
| --- | --- |
| `bench.py` | matmul and transformer-training throughput, PyTorch MPS vs MLX |
| `rangefetch.py` | parallel byte-range downloader (the Lakh tarball at ~4 MB/s instead of 30 KB/s) |
| `dmz_filter.py` | scores every MIDI file for the 2006 DMZ sound by what it plays: tempo, half-time / one-drop drums, offbeat skank, sparse sub, minor mode, density |
| `prep_finetune.py` | the files scoring >= a threshold, converted and tokenized as the model was trained (augmented, so it keeps its anticipation / infilling mode) |
| `finetune.py` | fine-tunes `music-medium-800k` on MPS (bf16 autocast, AdamW, cosine LR); `--probe` measures speed and memory |
| `amt_smoke.py`, `fast_smoke.py` | smoke tests: the reference sampler (~8 tok/s) vs the cached one (~87 tok/s) |
| `fast_sample.py` | a KV-cached sampler for the model: ~11x the reference's speed |
| `lead_core.py` | the constrained lead sampler: one instrument, a register, the song's mode, room between notes, optionally riddim's rhythm with only the pitches chosen |
| `lead_worker.py` | the process riddim talks to (`riddim/model_lead.py`): JSON lines in, leads out |
| `amt_lead.py`, `lead_stats.py` | A/B renders of riddim's lead vs the model's, and the numbers to compare them |

## The run

```bash
# data: Lakh MIDI, matched subset (CC-BY 4.0)
python rangefetch.py http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz $ML_HOME/data/lmd/lmd_matched.tar.gz 16 SIZE  # SIZE = Content-Length
python dmz_filter.py $ML_HOME/data/lmd/extracted $ML_HOME/out/dmz_scores.jsonl
python prep_finetune.py 4.0 $ML_HOME/data/ft_dmz          # ~3,750 files -> 266M tokens
python finetune.py $ML_HOME/data/ft_dmz $ML_HOME/models/ft_dmz 1500
```

On an M5 Pro (48 GB): the fine-tune runs at ~2,100 tok/s in 24 GB with batch 2
x 16 accumulation; validation loss 0.611 -> 0.518 by step 400.

The Last.fm tags for the Million Song Dataset are research-only and
non-commercial, so they are not used - not even to pick files. The filter
listens to the notes instead.

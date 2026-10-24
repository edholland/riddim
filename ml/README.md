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
| `build_data.py` | v2 data: role-normalised to riddim's band (lead 81, bass 38, chords 4, pad 89, drums), moved to riddim's tempos, transposed, deduplicated by tune; tokenized as the lead task (band as controls, lead as events - the inference task) and the model's general augmented task |
| `riddim_corpus.py` | riddim's own songs as general-task data: its band grammar, never its leads |
| `mlx_gpt2.py` | GPT-2 in MLX, weight-for-weight with Hugging Face's (loss matches PyTorch to 4 decimals; checkpoints load in PyTorch unchanged) |
| `train_mlx.py` | the MLX trainer: compiled step, weighted data mixture, fp32 master weights with bf16 compute, resumable - 5.5k tok/s, 2.5x the PyTorch loop |
| `run_overnight.sh` | runs the trainer under caffeinate and restarts it from the last checkpoint if it dies |
| `eval_leads.py` | models as lead writers on riddim songs, through riddim's own model_lead, scored and rendered |
| `pipeline.sh` | all of it, end to end, each stage resuming |
| `fast_sample.py` | a KV-cached sampler for the model: ~11x the reference's speed |
| `lead_core.py` | the constrained lead sampler: one instrument, a register, the song's mode, room between notes, optionally riddim's rhythm with only the pitches chosen |
| `lead_worker.py` | the process riddim talks to (`riddim/model_lead.py`): JSON lines in, leads out |
| `collect_refs.py` | reference recordings in (archive.org items, or anything yt-dlp reads), audio + a provenance manifest out (licence recorded; unlicensed items marked study-only) |
| `judge.py` | CLAP embeddings: FAD and closeness of renders against the references, and a zero-shot real-record-vs-machine probability (`pip install torch transformers`) |
| `amt_lead.py`, `lead_stats.py` | A/B renders of riddim's lead vs the model's, and the numbers to compare them |

## v2: the pipeline (`pipeline.sh`)

1. **Data**: Lakh MIDI full (178k files, CC-BY 4.0), scored by `dmz_filter.py`.
   Two tiers - score >= 3.5 (4,070 files) and 2.0-3.5 (23,175 files).
2. **Normalised** (`build_data.py`): one lead (the melody track), one bass, up
   to two comping parts, drums - mapped onto the instruments riddim plays
   them on; time moved to the nearest riddim tempo (at most 25%);
   transposed; versions of the same tune dropped (5,630 duplicates).
   12,327 songs survive: 78M tokens of lead task, 397M of general task.
3. **riddim's corpus** (`riddim_corpus.py`): 4,000 songs, 58M tokens.
4. **Training** (`train_mlx.py`): mixture 30% lead-top, 30% lead-broad,
   15%/15% general, 10% riddim; 32,000 steps of 8 x 1024 tokens; lr 1.5e-5
   cosine; from the first fine-tune (`ft_dmz/final`). `best` is chosen on the
   top tier's lead-task validation loss.
5. **Judged** (`eval_leads.py`) on riddim live tracks against riddim's own leads.

riddim picks `~/ml/models/v2/best` up automatically when it exists.

## The first run (v1)

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

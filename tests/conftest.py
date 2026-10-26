"""Shared finders for the critic's stories.

The critic tests need a song with a particular story - a bass that skips
the root under a fast harmony, where slowing the chords helps and a new
bass take does not. Which seed tells that story changes whenever the
engine's music improves, so the tests find one rather than pin it.
"""

import os
from functools import lru_cache

# The tests hear the synths: a sample library on this machine (samples/packs)
# must not change what they render. tests/test_library.py builds its own.
os.environ["RIDDIM_LIBRARY"] = "off"


@lru_cache(maxsize=None)
def wall_seed(profile="roots_dub", bpm=72) -> int:
    """A seed whose only finding is the sub warning, which the loop cures by
    the harmonic-rhythm twist and then stops at (the re-roll does not help)."""
    from riddim import critique
    from riddim.compose import compose
    from riddim.params import SongParams
    for seed in range(1, 600):
        song = compose(params=SongParams(profile=profile, seed=seed, bpm=bpm))
        fs = critique.critique(song)
        if len(fs) != 1 or "confirms" not in fs[0].message or fs[0].severity != "warning":
            continue
        _, steps = critique.revise(song)
        if len(steps) == 2 and steps[0].applied and not steps[1].applied:
            return seed
    raise RuntimeError("no seed tells the wall story")


@lru_cache(maxsize=None)
def clean_seed(profile="dubstep_140") -> int:
    """A seed the critic finds nothing in."""
    from riddim import critique
    from riddim.compose import compose
    for seed in range(1, 300):
        if not critique.critique(compose(profile, seed=seed)):
            return seed
    raise RuntimeError("no clean seed")

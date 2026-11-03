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
# Nor may the bass rhythms mined from the references (riddim/theory/cells.json):
# the critic's stories are found on the generator. test_transcribe.py fakes its own.
os.environ["RIDDIM_CELLS"] = "off"
# Nor the trained model's leads, the CLI's default where it is installed.
os.environ["RIDDIM_MODEL_LEAD"] = "off"


def _rule_dubstep():
    """Dubstep plays the bass as the hook (Ed's pick), a different idiom from
    the rules most of these tests pin (a lead in the drops, the sub on eighths
    and few notes) and use dubstep to exercise. They test the engine's default
    path; test_bass_hook_gives_the_peaks_to_the_bass tests that one."""
    from dataclasses import replace
    from riddim import profile
    profile.PROFILES["dubstep_140"] = replace(profile.PROFILES["dubstep_140"], bass_hook=False)


_rule_dubstep()


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

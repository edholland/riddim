"""The CLI surface: --check reports, --fix revises, batch ranks.

All three run the critic, and all three are deterministic for a given seed -
so the output is the contract, and gets pinned like anything else.
"""

import re

from riddim.cli import main


def test_check_prints_score_and_suggested_fixes(capsys):
    """The critic found a fast harmony the bass cannot follow and a peak that
    arrives after the loudest thing: the score, the findings, and the fix for
    each, all in the report."""
    rc = main(["compose", "--profile", "roots_dub", "--seed", "3", "--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "check: 0.70" in out
    assert "confirms the chord root" in out
    assert "try: bars_per_chord = 2.0" in out
    assert "peak at 0.99 arrives after 1.00" in out


def test_check_on_a_clean_song_says_clean(capsys):
    rc = main(["compose", "--profile", "roots_dub", "--seed", "5", "--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "check: 1.00" in out
    assert "clean" in out


def test_fix_prints_the_trail(capsys):
    """The loop applies the harmonic-rhythm twist (it helps), tries a re-roll
    of the wandering figure (it does not), says so, and stops. The trail is
    the honest record of all of that."""
    rc = main(["compose", "--profile", "roots_dub", "--seed", "3", "--fix"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fix: 2 steps" in out
    assert "bars_per_chord = 2.0" in out
    assert "0.70 -> 0.80" in out
    assert "rejected - score did not improve" in out


def test_fix_on_a_clean_song_says_so(capsys):
    rc = main(["compose", "--profile", "roots_dub", "--seed", "5", "--fix"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fix: already clean" in out
    assert "step" not in out.split("fix:")[1]


def test_batch_ranks_deterministically(capsys):
    args = ["batch", "--profile", "roots_dub", "--seed", "7", "--count", "8",
            "--top", "3"]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert "roots_dub: 8 songs, scored by the critic" in first
    # Same inputs, same ranking: the engine's rules vote, nobody else's.
    assert main(args) == 0
    assert capsys.readouterr().out == first
    scores = [float(m) for m in re.findall(r"score ([0-9.]+)", first)]
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)
    assert 0.0 <= min(scores) <= max(scores) <= 1.0


def test_batch_fix_revises_the_top_and_reranks(capsys):
    args = ["batch", "--profile", "roots_dub", "--seed", "1", "--count", "8",
            "--top", "8", "--fix"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "revising the top 8" in out
    # The generator is disciplined: most of the batch is already clean...
    assert "1.00 -> 1.00" in out
    # ...and where it is not, the loop reports the honest before -> after.
    assert "0.70 -> 0.80" in out
    after = out.split("ranked after fixing:")[1]
    scores = [float(m) for m in re.findall(r"score ([0-9.]+)", after)]
    assert len(scores) == 8
    assert scores == sorted(scores, reverse=True)

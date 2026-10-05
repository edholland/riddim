"""Voice leading: choose the voicing of each chord that moves least.

This is the single highest-leverage piece of theory for making chord parts
sound deliberate rather than random. Given a progression, picking inversions
so that voices creep by a semitone or two - instead of every chord jumping
back to root position - is most of what "these chords sound professional"
actually means.
"""

from __future__ import annotations

from .chord import Chord, voicings


def voicing_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    """Total semitone movement between two voicings.

    When the voice counts differ we match as many voices as we can pairwise
    from the bottom up and charge the leftovers a flat cost, so a 3-note to
    4-note move is still comparable.
    """
    a_sorted, b_sorted = sorted(a), sorted(b)
    pairs = min(len(a_sorted), len(b_sorted))
    cost = sum(abs(a_sorted[i] - b_sorted[i]) for i in range(pairs))
    cost += 6 * abs(len(a_sorted) - len(b_sorted))
    return cost


def lead(
    chords: list[Chord],
    low: int,
    high: int,
    centre: int | None = None,
    max_span: int = 24,
    first: Chord | None = None,
) -> list[Chord]:
    """Re-voice a progression for minimal movement inside [low, high].

    Greedy from the first chord, which is good enough for the 2-4 chord vamps
    this music uses and stays predictable - a full Viterbi pass over eight
    chords picks differently in ways that are hard to reason about on stage.
    """
    if not chords:
        return []
    if centre is None:
        centre = (low + high) // 2

    out: list[Chord] = []
    previous: tuple[int, ...] | None = first.pitches if first else None

    for chord in chords:
        candidates = voicings(chord, low, high, max_span=max_span)
        if not candidates:
            # Register window too tight for this chord; keep it unvoiced
            # rather than silently dropping a chord out of the progression.
            out.append(chord)
            previous = chord.pitches
            continue

        if previous is None:
            # Open on the voicing closest to the centre of the window.
            best = min(
                candidates,
                key=lambda c: abs(sum(c.pitches) / len(c.pitches) - centre),
            )
        else:
            prev = previous  # bind for the closure
            best = min(
                candidates,
                key=lambda c: (
                    voicing_distance(prev, c.pitches)
                    # Mild pull toward the centre breaks ties away from the edges.
                    + abs(sum(c.pitches) / len(c.pitches) - centre) * 0.25
                ),
            )
        out.append(best)
        previous = best.pitches
    return out


def common_tones(a: Chord, b: Chord) -> set[int]:
    """Pitch classes shared by two chords - the notes that can be held."""
    return a.pitch_classes() & b.pitch_classes()


def spread(chord: Chord, low: int, high: int) -> Chord:
    """Open a close voicing out across the register.

    Keeps the bottom note, pushes the rest up into octaves, which leaves a
    hole in the low-mid for the sub to occupy.
    """
    ps = sorted(chord.pitches)
    if len(ps) < 2:
        return chord
    out = [ps[0]]
    for p in ps[1:]:
        note = p
        while note < out[-1] + 5 and note + 12 <= high:
            note += 12
        out.append(note)
    if max(out) > high or min(out) < low:
        return chord
    return chord.with_pitches(out)

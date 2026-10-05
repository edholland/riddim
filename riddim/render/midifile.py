"""Write a Song to a standard MIDI file.

This exists so the tool is useful with nothing plugged in. Auditioning a
sketch in a DAW takes seconds; wiring up the Deluge and the rack takes
minutes, and the iteration loop is where the musical decisions actually get
made.

Drum notes are remapped to General MIDI on the way out so the file sounds
right in any player, while the Song itself keeps Deluge kit-row numbering.
"""

from __future__ import annotations

import mido

from ..compose import KIT_ROWS
from ..song import Song
from ..units import TICKS_PER_BAR

# Deluge kit row -> General MIDI percussion, for audition only.
GM_DRUMS = {
    KIT_ROWS["kick"]: 36,    # Bass Drum 1
    KIT_ROWS["snare"]: 38,   # Acoustic Snare
    KIT_ROWS["hat"]: 42,     # Closed Hi-Hat
    KIT_ROWS["rim"]: 37,     # Side Stick
    KIT_ROWS["perc"]: 39,    # Hand Clap
    KIT_ROWS["shaker"]: 70,  # Maracas
}

# Our tick grid is 24 per 16th, so 96 per quarter note.
TICKS_PER_BEAT = 96


def write(song: Song, path: str, gm_drums: bool = True) -> str:
    """Write `song` to `path`. Returns the path."""
    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=song.name, time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(song.bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    _append_markers(meta, song)
    midi.tracks.append(meta)

    for index, part in enumerate(song.parts):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=part.role, time=0))

        is_drums = part.output == "kit"
        channel = 9 if is_drums else min(15, max(0, part.channel - 1))

        events: list[tuple[int, int, str, int, int]] = []
        for note in part.sorted_notes():
            midi_note = (
                GM_DRUMS.get(note.midi, note.midi) if (is_drums and gm_drums)
                else note.midi
            )
            if not 0 <= midi_note <= 127:
                continue
            # Sort key puts note_off before note_on at the same tick, so a
            # repeated note retriggers cleanly instead of being cut short.
            events.append((note.start, 1, "on", midi_note, note.velocity))
            events.append((note.end, 0, "off", midi_note, 0))

        events.sort(key=lambda e: (e[0], e[1]))

        previous = 0
        for tick, _, kind, midi_note, velocity in events:
            delta = max(0, tick - previous)
            previous = tick
            track.append(
                mido.Message(
                    "note_on" if kind == "on" else "note_off",
                    note=midi_note,
                    velocity=velocity,
                    channel=channel,
                    time=delta,
                )
            )
        midi.tracks.append(track)

    midi.save(path)
    return path


def _append_markers(track, song: Song) -> None:
    """Section markers, so the structure is visible in the DAW."""
    if song.arrangement is None:
        return
    previous = 0
    for section in song.arrangement.sections:
        tick = section.start_bar * TICKS_PER_BAR
        track.append(
            mido.MetaMessage("marker", text=section.name, time=max(0, tick - previous))
        )
        previous = tick

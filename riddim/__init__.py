"""Theory-grounded dub and dubstep composition for the Synthstrom Deluge.

The package is layered so each piece is useful on its own:

    riddim.theory   pure music theory - scales, chords, voice leading,
                    rhythm, motif development. No MIDI, no hardware.
    riddim.arrange  song structure and the energy curve.
    riddim.profile  what makes roots dub different from 140 dubstep.
    riddim.compose  puts the above together into a Song.
    riddim.song     the note/part model - the boundary between theory and MIDI.
    riddim.render   MIDI file output, for auditioning with nothing plugged in.
    riddim.deluge   the AI SysEx protocol and the session-clip renderer.
"""

from .compose import compose
from .song import Song, Part, Note, Rig

__all__ = ["compose", "Song", "Part", "Note", "Rig"]

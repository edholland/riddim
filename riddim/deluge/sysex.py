"""The Deluge AI SysEx protocol.

A direct transcription of the command table in
`DelugeFirmware/src/deluge/io/midi/ai_sysex.cpp`. Every payload layout here
was read off that file's parser, including the length thresholds the firmware
uses to decide between 14-bit and 21-bit timing - those are load-bearing, so
the encoders below pick the wide form explicitly rather than by accident.

This module builds byte lists only. Sending them is `client.py`'s job.
"""

from __future__ import annotations

from enum import IntEnum

# Synthstrom manufacturer ID, Deluge device, AI namespace.
HEADER = [0x00, 0x21, 0x7B, 0x01, 0x06]


class Command(IntEnum):
    PING = 0
    SET_CURRENT_NOTE = 1
    CLEAR_CURRENT_CLIP = 2
    SELECT_SESSION_ROW = 3
    SET_ROW_EXPRESSION = 4
    SET_CURRENT_CLIP_LENGTH = 5
    SET_CLIP_PARAM_AUTOMATION = 6
    SELECT_SESSION_INDEX = 7
    CLONE_SESSION_INDEX = 8
    CREATE_SESSION_INDEX = 9
    CLAMP_SESSION_INDEX_LENGTH = 10
    SET_CURRENT_CLIP_NAME = 11
    LOAD_CURRENT_SYNTH_PRESET = 12
    SYNC_SHARED_OUTPUT_PARAMS = 13
    CLONE_SESSION_VISIBLE = 14
    CREATE_SESSION_VISIBLE = 15
    CLAMP_SESSION_VISIBLE_LENGTH = 16
    SET_SECTION_REPEAT = 17
    SET_SIDECHAIN = 18
    SAVE_SONG = 19
    LOAD_SONG = 20
    ARM_SECTION = 21
    SET_ROW_PARAM_AUTOMATION = 22
    LOAD_CURRENT_KIT_PRESET = 23


class Status(IntEnum):
    OK = 0
    BAD_LENGTH = 1
    NO_SONG = 2
    NO_INSTRUMENT_CLIP = 3
    NOTE_ROW_FAILED = 4
    ACTION_FAILED = 5
    ADD_FAILED = 6
    CLEAR_FAILED = 7
    NO_KIT_ROW = 8
    NO_CLIP = 9
    BAD_PARAM = 10
    SD_CARD_UNAVAILABLE = 11
    FILE_NOT_FOUND = 12
    FILE_EXISTS = 13
    BUSY = 14
    NEEDS_UI_SAVE = 15


class OutputType(IntEnum):
    SYNTH = 0
    KIT = 1
    MIDI = 2
    CV = 3
    AUDIO = 4


# SET_CLIP_PARAM_AUTOMATION / SET_ROW_PARAM_AUTOMATION param aliases, as
# understood by the firmware. The first four are every firmware's; the rest,
# and automation on kit clips at all, need branch ai-kit-automation. On a
# synth clip they move the synth; on a kit clip, the whole kit (its "affect
# entire" effects); on a kit row, that drum's own sound. decay and release
# are sounds' only (a synth or a row), not the whole kit's.
PARAMS = {"lpf": 0, "resonance": 1, "hpf": 2, "reverb": 3, "delay": 4,
          "volume": 5, "pan": 6, "pitch": 7, "decay": 8, "release": 9,
          "delay_rate": 10}

# SET_ROW_EXPRESSION dimensions.
EXPRESSION = {"pitch": 0, "timbre": 1, "pressure": 2}

MAX_14 = 0x3FFF
MAX_21 = 0x1FFFFF


# --------------------------------------------------------------------------
# Encoding primitives
# --------------------------------------------------------------------------

def enc14(value: int) -> list[int]:
    if not 0 <= value <= MAX_14:
        raise ValueError(f"value {value} does not fit in 14 bits")
    return [(value >> 7) & 0x7F, value & 0x7F]


def enc21(value: int) -> list[int]:
    if not 0 <= value <= MAX_21:
        raise ValueError(f"value {value} does not fit in 21 bits")
    return [(value >> 14) & 0x7F, (value >> 7) & 0x7F, value & 0x7F]


def dec21(b0: int, b1: int, b2: int) -> int:
    return ((b0 & 0x7F) << 14) | ((b1 & 0x7F) << 7) | (b2 & 0x7F)


def dec14(b0: int, b1: int) -> int:
    return ((b0 & 0x7F) << 7) | (b1 & 0x7F)


def scale_14(value: float) -> int:
    """Map a 0..1 float onto the firmware's 0..16383 parameter range."""
    return max(0, min(MAX_14, int(round(value * MAX_14))))


def ascii7(text: str, limit: int = 32) -> list[int]:
    """7-bit-safe bytes for a name. Anything outside ASCII is dropped."""
    return [ord(c) & 0x7F for c in text[:limit] if 32 <= ord(c) < 127]


def frame(payload: list[int]) -> list[int]:
    """Prepend the AI namespace header. Returns the SysEx data body.

    The F0/F7 wrapper is added by the MIDI library, not here.
    """
    for byte in payload:
        if not 0 <= byte <= 0x7F:
            raise ValueError(f"byte {byte} is not 7-bit safe")
    return HEADER + payload


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def ping() -> list[int]:
    return frame([Command.PING])


def set_note(note: int, position: int, length: int, velocity: int = 100) -> list[int]:
    """Write one note into the currently selected clip.

    Uses the 21-bit timing form whenever either value needs it. The firmware
    switches on payload length (>= 9 bytes means wide), so we must not pad.
    """
    note = max(0, min(127, int(note)))
    velocity = max(1, min(127, int(velocity)))
    length = max(1, int(length))
    if position > MAX_14 or length > MAX_14:
        body = [Command.SET_CURRENT_NOTE, note] + enc21(position) + enc21(length) + [velocity]
    else:
        body = [Command.SET_CURRENT_NOTE, note] + enc14(position) + enc14(length) + [velocity]
    return frame(body)


def clear_clip(start: int = 0, end: int = MAX_14) -> list[int]:
    """Erase notes in [start, end) of the current clip."""
    if start > MAX_14 or end > MAX_14:
        body = [Command.CLEAR_CURRENT_CLIP] + enc21(start) + enc21(end)
    else:
        body = [Command.CLEAR_CURRENT_CLIP] + enc14(start) + enc14(end)
    return frame(body)


def select_row(row: int, create: OutputType | None = None) -> list[int]:
    """Select a visible session row. 127 in the create slot means don't create."""
    create_byte = 127 if create is None else int(create)
    return frame([Command.SELECT_SESSION_ROW, row & 0x7F, create_byte])


def select_index(index: int) -> list[int]:
    """Select a session clip by absolute index."""
    return frame([Command.SELECT_SESSION_INDEX] + enc14(index))


def set_clip_length(length: int) -> list[int]:
    """Set the current clip's loop length in ticks."""
    if length > MAX_14:
        return frame([Command.SET_CURRENT_CLIP_LENGTH] + enc21(length))
    return frame([Command.SET_CURRENT_CLIP_LENGTH] + enc14(length))


def create_index(
    index: int,
    output: OutputType = OutputType.SYNTH,
    section: int = 0,
    length: int | None = None,
) -> list[int]:
    """Create a new session clip at an absolute index."""
    body = [Command.CREATE_SESSION_INDEX] + enc14(index) + [int(output), section & 0x7F]
    if length is not None:
        # 8-byte payload selects the 21-bit read in the firmware.
        body += enc21(length)
    return frame(body)


def create_visible(
    row: int,
    output: OutputType = OutputType.SYNTH,
    section: int = 0,
    length: int | None = None,
) -> list[int]:
    """Create a clip at a visible song row, counted from the top."""
    body = [Command.CREATE_SESSION_VISIBLE] + enc14(row) + [int(output), section & 0x7F]
    if length is not None:
        body += enc21(length)
    return frame(body)


def clone_index(
    source: int,
    dest: int,
    section: int | None = None,
    length: int | None = None,
) -> list[int]:
    body = [Command.CLONE_SESSION_INDEX] + enc14(source) + enc14(dest)
    if section is not None or length is not None:
        body += [(section or 0) & 0x7F]
    if length is not None:
        body += enc21(length)
    return frame(body)


def clamp_index_length(index: int, length: int) -> list[int]:
    return frame([Command.CLAMP_SESSION_INDEX_LENGTH] + enc14(index) + enc21(length))


def clamp_visible_length(row: int, length: int) -> list[int]:
    return frame([Command.CLAMP_SESSION_VISIBLE_LENGTH] + enc14(row) + enc21(length))


def set_clip_name(name: str, clip: bool = True, output: bool = True) -> list[int]:
    flags = (1 if clip else 0) | (2 if output else 0)
    return frame([Command.SET_CURRENT_CLIP_NAME, flags] + ascii7(name))


def load_synth_preset(preset: str) -> list[int]:
    return frame([Command.LOAD_CURRENT_SYNTH_PRESET] + ascii7(preset, limit=64))


def song_name(name: str, limit: int = 40) -> str:
    """A song name the firmware will accept as a file name: letters, digits,
    space, - and _ (anything else becomes _), at most `limit` characters."""
    out = "".join(c if (c.isascii() and (c.isalnum() or c in " -_")) else "_" for c in name)
    out = out.strip()[:limit]
    if not out:
        raise ValueError(f"no usable characters in song name {name!r}")
    return out


def save_song(name: str, overwrite: bool = False) -> list[int]:
    """Save the current song as SONGS/<name>.XML (custom firmware 19)."""
    return frame([Command.SAVE_SONG, 1 if overwrite else 0] + ascii7(song_name(name), limit=60))


def arm_section(start: int = 0, park: int = 23) -> list[int]:
    """Make `start` the section that plays when play is pressed; park empty
    clips (a template's) in section `park`, inactive (custom firmware 21)."""
    for v in (start, park):
        if not 0 <= v < 24:
            raise ValueError(f"section {v} out of range 0..23")
    return frame([Command.ARM_SECTION, start, park])


def load_song(name: str) -> list[int]:
    """Load SONGS/<name>.XML (custom firmware 20). The transport must be
    stopped and the Deluge on a main view, or it replies BUSY."""
    return frame([Command.LOAD_SONG] + ascii7(song_name(name), limit=60))


def sync_shared_output() -> list[int]:
    return frame([Command.SYNC_SHARED_OUTPUT_PARAMS])


def set_param_automation(
    param: str,
    start: int,
    length: int,
    start_value: float,
    end_value: float | None = None,
) -> list[int]:
    """Write a filter/reverb hold or ramp into the current clip.

    `start_value` and `end_value` are 0..1; shape 1 (ramp) is selected
    automatically when an end value is supplied.
    """
    if param not in PARAMS:
        raise ValueError(
            f"unknown param {param!r}; known: {', '.join(sorted(PARAMS))}"
        )
    shape = 1 if end_value is not None else 0
    start_raw = scale_14(start_value)
    end_raw = scale_14(end_value if end_value is not None else start_value)

    head = [Command.SET_CLIP_PARAM_AUTOMATION, PARAMS[param], shape]
    if start > MAX_14 or length > MAX_14:
        body = head + enc21(start) + enc21(length) + enc14(start_raw) + enc14(end_raw)
    else:
        body = head + enc14(start) + enc14(length) + enc14(start_raw) + enc14(end_raw)
    return frame(body)


def set_row_param_automation(
    note: int,
    param: str,
    start: int,
    length: int,
    start_value: float,
    end_value: float | None = None,
) -> list[int]:
    """Automate one drum of the current kit clip (custom firmware 22):
    `note` is the kit row (36 the bottom row), the rest as
    `set_param_automation` - a hold, or a ramp when `end_value` is given."""
    if param not in PARAMS:
        raise ValueError(
            f"unknown param {param!r}; known: {', '.join(sorted(PARAMS))}"
        )
    if not 0 <= note <= 127:
        raise ValueError(f"note {note} out of range")
    shape = 1 if end_value is not None else 0
    start_raw = scale_14(start_value)
    end_raw = scale_14(end_value if end_value is not None else start_value)
    head = [Command.SET_ROW_PARAM_AUTOMATION, note & 0x7F, PARAMS[param], shape]
    if start > MAX_14 or length > MAX_14:
        body = head + enc21(start) + enc21(length) + enc14(start_raw) + enc14(end_raw)
    else:
        body = head + enc14(start) + enc14(length) + enc14(start_raw) + enc14(end_raw)
    return frame(body)


def load_kit_preset(preset: str) -> list[int]:
    """Load KITS/<preset>.XML onto the current kit clip (custom firmware 23)."""
    return frame([Command.LOAD_CURRENT_KIT_PRESET] + ascii7(preset, limit=64))


def set_row_expression(
    note: int,
    dimension: str,
    start: int,
    length: int,
    value: float,
) -> list[int]:
    """Write per-row expression. For pitch, 0.5 is centre.

    Useful for portamento-style slides on a bass row, which is exactly the
    gesture dub bass wants and which a plain note list cannot express.
    """
    if dimension not in EXPRESSION:
        raise ValueError(
            f"unknown dimension {dimension!r}; known: {', '.join(sorted(EXPRESSION))}"
        )
    raw = scale_14(value)
    head = [Command.SET_ROW_EXPRESSION, note & 0x7F, EXPRESSION[dimension]]
    if start > MAX_14 or length > MAX_14:
        body = head + enc21(start) + enc21(length) + enc14(raw)
    else:
        body = head + enc14(start) + enc14(length) + enc14(raw)
    return frame(body)


def set_section_repeat(
    section: int,
    repeats: int = 0,
    loop: bool = False,
    loop_start: int | None = None,
    loop_end: int | None = None,
) -> list[int]:
    """Set a section's repeat count. 0 means infinite, 1 means play once.

    This is what turns a grid of session clips into an actual song: set the
    repeat count per section and enable the AI section loop, and the Deluge
    plays the arrangement start to finish on its own.
    """
    encoded = max(0, min(MAX_14, repeats + 8192))
    body = [Command.SET_SECTION_REPEAT, section & 0x7F] + enc14(encoded)
    body += [1 if loop else 0]
    if loop_start is not None:
        body += [loop_start & 0x7F]
        if loop_end is not None:
            body += [loop_end & 0x7F]
    return frame(body)


def set_sidechain_send(note: int, amount: float = 1.0) -> list[int]:
    """Make a kit row send to the sidechain. Note 36 is the bottom row."""
    return frame(
        [Command.SET_SIDECHAIN, 0, note & 0x7F] + enc14(scale_14(amount))
    )


def set_sidechain_receive(amount: float = 0.35) -> list[int]:
    """Make the current clip duck. 0.5 is neutral, lower ducks harder."""
    return frame([Command.SET_SIDECHAIN, 1] + enc14(scale_14(amount)))


# --------------------------------------------------------------------------
# Replies
# --------------------------------------------------------------------------

def parse_reply(data) -> tuple[int, Status, list[int]] | None:
    """Decode an AI SysEx reply into (command, status, extra bytes)."""
    body = list(data)
    if len(body) < 7 or body[:5] != HEADER:
        return None
    try:
        status = Status(body[6])
    except ValueError:
        status = body[6]
    return body[5], status, body[7:]

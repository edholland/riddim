"""Tests for the Deluge AI SysEx encoders.

These pin the wire format against the firmware parser in
`DelugeFirmware/src/deluge/io/midi/ai_sysex.cpp`. The payload *lengths* matter
as much as the contents: the firmware switches between 14-bit and 21-bit
timing by counting bytes, so an off-by-one here silently misreads every
position in the clip.
"""

import pytest

from riddim.deluge import sysex
from riddim.deluge.sysex import Command, OutputType, Status


def payload(message):
    """Strip the AI namespace header, leaving what the firmware sees as `data`."""
    assert message[:5] == sysex.HEADER
    return message[5:]


def test_all_bytes_are_seven_bit():
    messages = [
        sysex.ping(),
        sysex.set_note(127, 16383, 16383, 127),
        sysex.set_note(60, 100000, 100000, 100),
        sysex.clear_clip(0, 16383),
        sysex.set_clip_length(36864),
        sysex.create_index(9999, OutputType.MIDI, 11, 6144),
        sysex.set_param_automation("lpf", 0, 6144, 0.0, 1.0),
        sysex.set_clip_name("riddim"),
        sysex.set_section_repeat(3, 1, True, 0, 7),
        sysex.set_sidechain_send(36, 1.0),
    ]
    for message in messages:
        assert all(0 <= b <= 0x7F for b in message)


def test_encoding_round_trips():
    for value in (0, 1, 127, 128, 8191, 16383):
        assert sysex.dec14(*sysex.enc14(value)) == value
    for value in (0, 16384, 36864, 0x1FFFFF):
        assert sysex.dec21(*sysex.enc21(value)) == value


def test_encoders_reject_overflow():
    with pytest.raises(ValueError):
        sysex.enc14(16384)
    with pytest.raises(ValueError):
        sysex.enc21(0x200000)


def test_set_note_narrow_form_is_seven_bytes():
    # Firmware: `bool wideTiming = len >= 9`, so the narrow form must be < 9.
    body = payload(sysex.set_note(60, 384, 96, 100))
    assert len(body) == 7
    assert body[0] == Command.SET_CURRENT_NOTE
    assert body[1] == 60
    assert sysex.dec14(body[2], body[3]) == 384
    assert sysex.dec14(body[4], body[5]) == 96
    assert body[6] == 100


def test_set_note_wide_form_is_nine_bytes():
    body = payload(sysex.set_note(60, 20000, 96, 100))
    assert len(body) == 9
    assert sysex.dec21(body[2], body[3], body[4]) == 20000
    assert sysex.dec21(body[5], body[6], body[7]) == 96
    assert body[8] == 100


def test_set_note_switches_form_exactly_at_the_boundary():
    assert len(payload(sysex.set_note(60, sysex.MAX_14, 1, 100))) == 7
    assert len(payload(sysex.set_note(60, sysex.MAX_14 + 1, 1, 100))) == 9


def test_set_note_clamps_into_legal_ranges():
    body = payload(sysex.set_note(200, 0, 0, 0))
    assert body[1] <= 127
    assert body[6] >= 1          # firmware rejects velocity 0
    assert sysex.dec14(body[4], body[5]) >= 1


def test_clear_clip_form_lengths():
    # Firmware: `len < 5` rejected, `wideTiming = len >= 7`.
    assert len(payload(sysex.clear_clip(0, 6144))) == 5
    assert len(payload(sysex.clear_clip(0, 36864))) == 7


def test_clip_length_form_lengths():
    # Firmware: `len >= 4` selects read21.
    assert len(payload(sysex.set_clip_length(6144))) == 3
    assert len(payload(sysex.set_clip_length(36864))) == 4
    body = payload(sysex.set_clip_length(36864))
    assert sysex.dec21(body[1], body[2], body[3]) == 36864


def test_create_index_layout():
    # Firmware: `len < 5` rejected; loop length read as 21-bit when len >= 8.
    body = payload(sysex.create_index(5, OutputType.MIDI, 2, 6144))
    assert len(body) == 8
    assert body[0] == Command.CREATE_SESSION_INDEX
    assert sysex.dec14(body[1], body[2]) == 5
    assert body[3] == OutputType.MIDI
    assert body[4] == 2
    assert sysex.dec21(body[5], body[6], body[7]) == 6144


def test_create_index_without_length_is_minimal():
    assert len(payload(sysex.create_index(0, OutputType.SYNTH, 0))) == 5


def test_clone_index_carries_section_and_length():
    # Firmware: section read when len >= 6, 21-bit length when len >= 9.
    body = payload(sysex.clone_index(1, 9, section=3, length=6144))
    assert len(body) == 9
    assert sysex.dec14(body[1], body[2]) == 1
    assert sysex.dec14(body[3], body[4]) == 9
    assert body[5] == 3
    assert sysex.dec21(body[6], body[7], body[8]) == 6144


def test_automation_form_lengths():
    # Firmware: `len < 11` rejected, `wideTiming = len >= 13`.
    assert len(payload(sysex.set_param_automation("lpf", 0, 6144, 0.5))) == 11
    assert len(payload(sysex.set_param_automation("lpf", 0, 36864, 0.5))) == 13


def test_automation_shape_flag():
    hold = payload(sysex.set_param_automation("hpf", 0, 384, 0.5))
    ramp = payload(sysex.set_param_automation("hpf", 0, 384, 0.0, 1.0))
    assert hold[2] == 0
    assert ramp[2] == 1
    assert sysex.dec14(ramp[7], ramp[8]) == 0
    assert sysex.dec14(ramp[9], ramp[10]) == sysex.MAX_14


def test_automation_rejects_unknown_param():
    with pytest.raises(ValueError):
        sysex.set_param_automation("wavefolder", 0, 384, 0.5)


def test_row_expression_form_lengths():
    # Firmware: `len < 9` rejected, `wideTiming = len >= 11`.
    assert len(payload(sysex.set_row_expression(36, "pitch", 0, 384, 0.5))) == 9
    assert len(payload(sysex.set_row_expression(36, "pitch", 0, 36864, 0.5))) == 11


def test_section_repeat_offsets_by_8192():
    body = payload(sysex.set_section_repeat(2, repeats=1))
    assert body[1] == 2
    assert sysex.dec14(body[2], body[3]) == 8193   # signed14 + 8192
    assert body[4] == 0                            # loop disabled

    looped = payload(sysex.set_section_repeat(0, 1, loop=True, loop_start=0, loop_end=7))
    assert looped[4] == 1
    assert looped[5] == 0
    assert looped[6] == 7


def test_infinite_repeat_encodes_as_zero():
    body = payload(sysex.set_section_repeat(0, repeats=0))
    assert sysex.dec14(body[2], body[3]) == 8192


def test_clip_name_flags_and_ascii():
    body = payload(sysex.set_clip_name("sub3", clip=True, output=False))
    assert body[1] == 1
    assert bytes(body[2:]).decode() == "sub3"
    assert payload(sysex.set_clip_name("x", clip=False, output=True))[1] == 2


def test_name_strips_non_ascii():
    body = payload(sysex.set_clip_name("dubéé"))
    assert all(32 <= b < 127 for b in body[2:])


def test_scale_14_endpoints():
    assert sysex.scale_14(0.0) == 0
    assert sysex.scale_14(1.0) == sysex.MAX_14
    assert sysex.scale_14(2.0) == sysex.MAX_14      # clamps
    assert sysex.scale_14(-1.0) == 0


def test_parse_reply():
    message = sysex.HEADER + [Command.PING, Status.OK]
    assert sysex.parse_reply(message) == (Command.PING, Status.OK, [])

    with_extra = sysex.HEADER + [Command.CLAMP_SESSION_INDEX_LENGTH, Status.OK, 1, 2]
    command, status, extra = sysex.parse_reply(with_extra)
    assert command == Command.CLAMP_SESSION_INDEX_LENGTH
    assert status == Status.OK
    assert extra == [1, 2]


def test_parse_reply_ignores_foreign_sysex():
    assert sysex.parse_reply([0x7E, 0x00, 0x06, 0x01]) is None
    assert sysex.parse_reply(sysex.HEADER) is None       # too short


def test_frame_rejects_non_seven_bit_payload():
    with pytest.raises(ValueError):
        sysex.frame([0xF0])

"""Drums on the Deluge: the riddim kits, the kit preset load, and the desk
moves written into kit clips (firmware branch ai-kit-automation)."""

import os
import wave
import xml.etree.ElementTree as ET

import pytest

from riddim import profile as profile_mod
from riddim.compose import KIT_ROWS, compose
from riddim.deluge import render, sysex
from riddim.units import TICKS_PER_BAR

from test_render import FakeDeluge


# ---------------------------------------------------------------- protocol

def test_row_automation_bytes():
    """SET_ROW_PARAM_AUTOMATION as the firmware parses it: row note, alias,
    shape, 14-bit timing (21-bit past 16383), then two 14-bit values."""
    msg = sysex.set_row_param_automation(37, "delay", 96, 768, 0.3, 0.7)
    body = msg[5:]
    assert body[0] == sysex.Command.SET_ROW_PARAM_AUTOMATION == 22
    assert body[1:4] == [37, sysex.PARAMS["delay"], 1]
    assert sysex.dec14(body[4], body[5]) == 96 and sysex.dec14(body[6], body[7]) == 768
    assert sysex.dec14(body[8], body[9]) == sysex.scale_14(0.3)
    assert sysex.dec14(body[10], body[11]) == sysex.scale_14(0.7)
    assert len(body) == 12
    wide = sysex.set_row_param_automation(36, "hpf", 20000, 384, 0.5)[5:]
    assert len(wide) == 14 and wide[3] == 0
    assert sysex.dec21(*wide[4:7]) == 20000
    with pytest.raises(ValueError):
        sysex.set_row_param_automation(36, "wobble", 0, 1, 0.5)


def test_new_aliases_keep_the_old_ones():
    """The first four aliases are every firmware's and must not move."""
    assert [sysex.PARAMS[k] for k in ("lpf", "resonance", "hpf", "reverb")] == [0, 1, 2, 3]
    assert sorted(sysex.PARAMS.values()) == list(range(len(sysex.PARAMS)))


def test_kit_preset_bytes():
    body = sysex.load_kit_preset("RIDDIM DUB")[5:]
    assert body[0] == sysex.Command.LOAD_CURRENT_KIT_PRESET == 23
    assert bytes(body[1:]).decode() == "RIDDIM DUB"


# ---------------------------------------------------------------- compose

def test_kit_parts_carry_the_family_kit_and_its_desk_moves():
    """Every kit part names its family's kit; the drums get the build sweep
    and the backbeat's echo running into each drop, per drum where it is a
    channel's move (the snare row) and whole-kit where it is the desk's."""
    for name in ("roots_dub", "dubstep_140", "uk_garage"):
        song = compose(name, seed=3)
        family = profile_mod.get(name).family
        kit_parts = [p for p in song.parts if p.output == "kit"]
        assert kit_parts
        for p in kit_parts:
            assert p.preset == profile_mod.KIT_PRESETS[family]
        drums = song.part("drums")
        sweeps = [a for a in drums.automation if a.param == "hpf" and a.row is None]
        echo = [a for a in drums.automation if a.row == KIT_ROWS["snare"]]
        assert sweeps, name
        assert echo, name
        for a in sweeps:
            assert a.length == 2 * TICKS_PER_BAR and a.end_value > a.start_value
            # ...and it ends where a section ends: the drop is clean.
            ends = {s.start_tick + s.length_ticks for s in song.arrangement.sections}
            assert a.start + a.length in ends


def test_dub_throws_open_the_backbeat_echo():
    for seed in range(1, 12):
        song = compose("roots_dub", seed=seed)
        throws = [s for s in song.arrangement.sections if s.dub_throw and s.plays("drums")]
        if not throws:
            continue
        drums = song.part("drums")
        for s in throws:
            assert any(a.row == KIT_ROWS["snare"] and a.param == "delay"
                       and a.start == s.start_tick and a.end_value is None
                       for a in drums.automation)
        return
    pytest.skip("no dub throw in these seeds")


# ---------------------------------------------------------------- send

def test_send_with_kits_loads_once_and_clones_from_the_loaded_clip():
    song = compose("roots_dub", seed=3)
    fake = FakeDeluge(clips=6)
    rows = {"sub": 1, "chords": 2, "lead": 3, "pad": 4, "drums": 5, "perc": 6}
    render.send(song, fake, template_rows=rows, progress=False, kits=True)
    loads = fake.commands_of(sysex.Command.LOAD_CURRENT_KIT_PRESET)
    kit_roles = [p.role for p in song.parts if p.output == "kit"]
    assert len(loads) == len(kit_roles)
    assert all(bytes(c[6:]).decode() == "RIDDIM DUB" for c in loads)
    rows_written = fake.commands_of(sysex.Command.SET_ROW_PARAM_AUTOMATION)
    assert rows_written
    # every drums clip after the first clones the first generated drums clip
    template_indices = set(render.rows_to_indices(rows, 6).values())
    clones = [src for dest, src in sorted(fake.cloned_from.items())]
    assert any(src not in template_indices for src in clones)


def test_send_without_kits_is_what_old_firmware_saw():
    song = compose("roots_dub", seed=3)
    fake = FakeDeluge()
    render.send(song, fake, progress=False)
    assert not fake.commands_of(sysex.Command.LOAD_CURRENT_KIT_PRESET)
    assert not fake.commands_of(sysex.Command.SET_ROW_PARAM_AUTOMATION)


# ---------------------------------------------------------------- the kit files

def test_knob_and_cable_encoding_match_the_firmware():
    kit = pytest.importorskip("riddim.deluge.kit")
    # getParamFromUserValue(GLOBAL_VOLUME_POST_FX, 40), and the default
    # velocity->volume cable at 50 (Sound::setupAsDefaultSynth).
    assert kit.knob(40) == "0x4CCCCCA8"
    assert kit.knob(0) == "0x80000000"
    assert kit.knob(50) == "0x7FFFFFD2"
    assert kit.cable(50) == "0x3FFFFFE8"
    assert kit.knob(kit.pan_knob(0.0)) in ("0xFFFFFFE9", "0x00000000")


def test_kit_export(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from riddim.deluge import kit
    paths = kit.export(str(tmp_path), ("dub",), progress=None)
    xml_path = tmp_path / "KITS" / "RIDDIM DUB.XML"
    assert str(xml_path) in paths
    root = ET.parse(xml_path).getroot()
    assert root.tag == "kit"
    assert root.find("defaultParams") is not None     # the kit's own params: required
    sounds = root.find("soundSources").findall("sound")
    # bottom row first, in riddim's row order
    order = [n for n, _ in sorted(KIT_ROWS.items(), key=lambda kv: kv[1])]
    assert [s.get("name") for s in sounds] == [n.upper() for n in order]
    for s in sounds:
        osc = s.find("osc1")
        f = tmp_path / osc.get("fileName")
        assert f.exists(), f
        with wave.open(str(f)) as w:
            assert w.getnchannels() == 1 and w.getframerate() == 44100
            assert int(osc.find("zone").get("endSamplePos")) == w.getnframes()
        params = s.find("defaultParams")
        for attr in ("lpfFrequency", "volume", "pan", "reverbAmount", "delayFeedback"):
            assert params.get(attr), attr        # nothing left to an undefined default
        assert params.find("envelope1") is not None
        cables = {(c.get("source"), c.get("destination")) for c in params.iter("patchCable")}
        assert ("velocity", "volume") in cables
    by = {s.get("name"): s for s in sounds}
    assert by["HAT"].find("osc1").get("loopMode") == "0"      # CUT: the note closes it
    assert by["RISER"].find("osc1").get("loopMode") == "3"    # STRETCH: fits any build
    assert by["KICK"].get("sideChainSend")
    # dub's backbeat goes to the echo
    assert by["SNARE"].find("defaultParams").get("delayFeedback") != "0x80000000"
    assert os.path.getsize(xml_path) > 1000


def test_a_kit_that_does_not_load_stops_the_send():
    """Without its kit a part plays the template's sounds on riddim's rows:
    the send stops and says why, instead of writing on."""
    from riddim.deluge.client import DelugeError
    song = compose("roots_dub", seed=3)
    fake = FakeDeluge(fail_on={sysex.Command.LOAD_CURRENT_KIT_PRESET}, clips=6)
    rows = {"sub": 1, "chords": 2, "lead": 3, "pad": 4, "drums": 5, "perc": 6}
    with pytest.raises(DelugeError, match="RIDDIM DUB.*did not load"):
        render.send(song, fake, template_rows=rows, progress=False, kits=True)


def test_kit_failure_says_what_to_do():
    from riddim.deluge.client import Reply
    msg = render.kit_failure("RIDDIM DUB", Reply(23, sysex.Status.FILE_NOT_FOUND, []), "drums")
    assert "KITS/RIDDIM DUB.XML" in msg and "deluge-kits" in msg and "card" in msg
    assert "flash" in render.kit_failure("X", Reply(23, sysex.Status.BAD_LENGTH, []))


def test_install_kits_in_template():
    """Load the template, keep an untouched copy, load the kit on its drums
    and perc clips, save it over itself."""
    fake = FakeDeluge(clips=6)
    rows = {"sub": 1, "chords": 2, "lead": 3, "pad": 4, "drums": 5, "perc": 6}
    log = []
    render.install_kits_in_template(fake, "Template", rows, "RIDDIM DUB", log=log.append)
    order = [e for e, _ in fake.log if e in (sysex.Command.LOAD_SONG, sysex.Command.SAVE_SONG,
                                             sysex.Command.LOAD_CURRENT_KIT_PRESET)]
    assert order == [sysex.Command.LOAD_SONG, sysex.Command.SAVE_SONG,
                     sysex.Command.LOAD_CURRENT_KIT_PRESET, sysex.Command.LOAD_CURRENT_KIT_PRESET,
                     sysex.Command.SAVE_SONG]
    saves = fake.commands_of(sysex.Command.SAVE_SONG)
    assert bytes(saves[0][7:]).decode() == "Template BACKUP" and saves[0][6] == 0
    assert bytes(saves[1][7:]).decode() == "Template" and saves[1][6] == 1
    # the kits went onto the template's drums (row 5) and perc (row 6) clips
    selects = [sysex.dec14(*c[6:8]) for c in fake.commands_of(sysex.Command.SELECT_SESSION_INDEX)]
    idx = render.rows_to_indices({"drums": 5, "perc": 6}, 6)
    assert idx["drums"] in selects and idx["perc"] in selects


def test_cli_sends_with_kits_by_default():
    from riddim.cli import build_parser
    args = build_parser().parse_args(["send", "--dry-run"])
    assert args.kits is True
    assert build_parser().parse_args(["send", "--no-kits"]).kits is False


def test_kits_refused_on_an_older_firmware_build():
    """A build that answers PING without the kit protocol version cannot
    load the kits right: refuse before writing anything."""
    from riddim.deluge.client import DelugeError, Reply
    song = compose("roots_dub", seed=3)

    def fake_with(extra):
        fake = FakeDeluge(clips=6)
        fake.dry_run = False                  # so ping() reads the version
        real = fake.send

        def send(data, expect=None):
            if expect == sysex.Command.PING:
                return Reply(expect, sysex.Status.OK, extra)
            return real(data, expect)
        fake.send = send
        fake._in = object()                   # a reply port is there
        return fake

    old = fake_with([])
    with pytest.raises(DelugeError, match="older firmware build"):
        render.send(song, old, progress=False, kits=True)
    assert not old.commands_of(sysex.Command.CREATE_SESSION_INDEX)   # nothing written
    new = fake_with([render.KIT_PROTOCOL])
    new.ping()
    assert new.protocol == render.KIT_PROTOCOL
    render._check_kit_firmware(new)          # passes

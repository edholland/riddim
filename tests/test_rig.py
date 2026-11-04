import json

import numpy as np
import pytest

from riddim import rig
from riddim.cli import main


def test_probe_hears_the_note_on_the_routed_input_only():
    _, reports = rig.probe(rig.SimRig(routes={3: 2}), channel=3, note=36)
    assert [r.channel for r in reports if r.sounding] == [3]
    heard = reports[2]
    assert heard.note == "C2"
    assert abs(heard.cents) < 10
    assert heard.latency_ms == pytest.approx(12, abs=3)
    assert not heard.clipped


@pytest.mark.parametrize("note,name", [(28, "E1"), (36, "C2"), (48, "C3"), (69, "A4"), (84, "C6")])
def test_pitch_across_the_range_a_bass_and_lead_use(note, name):
    _, reports = rig.probe(rig.SimRig(), note=note)
    assert reports[1].note == name


def test_a_silent_rig_is_reported_silent():
    _, reports = rig.probe(rig.SimRig(routes={}))
    assert not any(r.sounding for r in reports)
    assert "nothing sounded" in rig.describe(reports)


def test_noise_is_not_read_as_pitch():
    rate = 48000
    x = np.random.default_rng(1).standard_normal(rate).astype(np.float32)
    assert rig.estimate_f0(x, rate) is None


def test_clipping_is_flagged():
    _, reports = rig.probe(rig.SimRig(level=4.0))
    take = rig.Take(np.clip(rig.SimRig(level=4.0).record(
        rig.note_events(1, 36, 127, 0.5, 1.0), 2.0).audio, -1, 1), 48000, [(0.5, (0x90, 36, 127))])
    assert rig.analyse(take)[1].clipped


def test_scan_learns_the_patching():
    routes = {1: 0, 2: 1, 5: 1}
    result = rig.scan(rig.SimRig(routes=routes), midi_channels=range(1, 7))
    assert result.routes == {1: [1], 2: [2], 5: [2]}
    assert result.silent == [3, 4, 6]


def test_a_take_always_cleans_up_after_itself():
    assert rig.silence_events([1, 16]) == [(0xB0, 123, 0), (0xBF, 123, 0)]


def test_the_macs_own_mic_is_never_picked_by_accident(monkeypatch):
    monkeypatch.setattr(rig, "audio_inputs", lambda: [
        {"index": 1, "name": "MacBook Pro Microphone", "channels": 1, "rate": 48000.0}])
    with pytest.raises(rig.RigError, match="MacBook Pro Microphone"):
        rig.find_input(None)


def test_saved_take_has_stems_for_what_sounded(tmp_path):
    take, reports = rig.probe(rig.SimRig())
    rig.save_take(take, reports, tmp_path)
    assert [p.name for p in tmp_path.glob("*.wav")] == ["in02.wav"]
    saved = json.loads((tmp_path / "report.json").read_text())
    assert len(saved["channels"]) == 4
    assert saved["sent"][0]["data"][0] == 0x90


def test_cli_probe_and_scan_run_on_the_sim(tmp_path, capsys):
    assert main(["rig", "probe", "--sim", "--out", str(tmp_path / "t")]) == 0
    assert (tmp_path / "t" / "report.json").exists()
    assert "in2" in capsys.readouterr().out
    assert main(["rig", "scan", "--sim", "--midi-channels", "1-2"]) == 0
    assert "MIDI 1 -> in2" in capsys.readouterr().out

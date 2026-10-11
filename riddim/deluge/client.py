"""Talking to the Deluge over MIDI SysEx.

Thin and synchronous on purpose. Every command that can fail waits for the
firmware's status reply and retries a couple of times, because USB MIDI drops
messages under load and a silently missing note in bar 43 is much harder to
debug than a loud failure here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mido

from . import sysex
from .sysex import Status


class DelugeError(RuntimeError):
    pass


def find_port(names: list[str], hint: str | None = None) -> str:
    """Pick a MIDI port. Prefers an explicit hint, then anything 'Deluge'."""
    if not names:
        raise DelugeError(
            "no MIDI ports found - is the Deluge connected and powered on?"
        )
    if hint:
        for name in names:
            if hint.lower() in name.lower():
                return name
        raise DelugeError(
            f"no MIDI port matching {hint!r}; available: {', '.join(names)}"
        )
    for name in names:
        if "deluge" in name.lower():
            return name
    return names[0]


@dataclass
class Reply:
    command: int
    status: Status | int
    extra: list

    @property
    def ok(self) -> bool:
        return self.status == Status.OK

    def length_detail(self) -> dict | None:
        """Decode a CLAMP_*_LENGTH reply.

        The firmware answers these with the clip's resulting state, which is
        the only way to read a clip's length back - there is no pure getter.
        `max_row_length` is the one that matters: a note row carrying its own
        `loopLengthIfIndependent` loops at a different length from its clip,
        which is how a clip ends up behaving as though it were far longer than
        it was set to.
        """
        if len(self.extra) < 13:
            return None
        e = self.extra
        return {
            "index": sysex.dec21(e[0], e[1], e[2]),
            "loop_length": sysex.dec21(e[3], e[4], e[5]),
            "max_length": sysex.dec21(e[6], e[7], e[8]),
            "max_row_length": sysex.dec21(e[9], e[10], e[11]),
            "rows": e[12],
        }


class Deluge:
    """A connection to one Deluge.

    Use as a context manager::

        with Deluge() as d:
            d.ping()
    """

    def __init__(
        self,
        output: str | None = None,
        input: str | None = None,
        timeout: float = 1.0,
        retries: int = 3,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
        self.dry_run = dry_run
        self.sent = 0
        self._out = None
        self._in = None
        self._output_hint = output
        self._input_hint = input if input is not None else output

    # -- connection ------------------------------------------------------

    def open(self) -> "Deluge":
        if self.dry_run:
            return self
        out_name = find_port(mido.get_output_names(), self._output_hint)
        self._out = mido.open_output(out_name)
        try:
            in_name = find_port(mido.get_input_names(), self._input_hint)
            self._in = mido.open_input(in_name)
        except DelugeError:
            # Output-only still works; we just cannot confirm anything.
            self._in = None
        if self.verbose:
            print(f"deluge: out={out_name!r} in={self._in and self._in.name!r}")
        self._warm_up()
        return self

    def _warm_up(self) -> None:
        """Absorb the first message, which is reliably dropped.

        Observed on every Deluge USB port: the first SysEx sent after opening
        a port never gets a reply, while an identical second one does. Without
        this, the first real command of a send silently goes missing - and
        since that command is usually the one creating clip 0, everything
        after it lands at the wrong index.
        """
        if self._in is None:
            return
        for _ in range(2):
            if self.send(sysex.ping(), sysex.Command.PING) is not None:
                return

    def close(self) -> None:
        for port in (self._out, self._in):
            if port is not None:
                port.close()
        self._out = self._in = None

    def __enter__(self) -> "Deluge":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- transport -------------------------------------------------------

    def send(self, data: list[int], expect: int | None = None) -> Reply | None:
        """Send one SysEx message and wait for its reply."""
        self.sent += 1
        if self.verbose:
            head = " ".join(f"{b:02X}" for b in data[:12])
            print(f"  -> {head}{'...' if len(data) > 12 else ''}")
        if self.dry_run:
            return Reply(expect or 0, Status.OK, [])

        if self._in is not None:
            for _ in self._in.iter_pending():
                pass  # drain stale replies

        self._out.send(mido.Message("sysex", data=data))
        if self._in is None:
            time.sleep(0.002)
            return None
        return self._await(expect)

    def _await(self, expect: int | None) -> Reply | None:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            for msg in self._in.iter_pending():
                if msg.type != "sysex":
                    continue
                parsed = sysex.parse_reply(msg.data)
                if parsed is None:
                    continue
                command, status, extra = parsed
                if expect is not None and command != expect:
                    continue  # stale reply from an earlier command
                return Reply(command, status, extra)
            time.sleep(0.001)
        return None

    def command(self, data: list[int], expect: int, label: str = "") -> bool:
        """Send with retries. Returns True if the firmware replied OK.

        A missing reply is treated as a failure worth retrying, but not as a
        hard error - plenty of setups run output-only.
        """
        for attempt in range(1, self.retries + 1):
            reply = self.send(data, expect)
            if reply is None:
                return self._in is None  # no input port: assume it landed
            if reply.ok:
                return True
            if self.verbose or attempt == self.retries:
                name = getattr(reply.status, "name", reply.status)
                print(f"  !! {label or expect}: {name}")
            time.sleep(0.05)
        return False

    def slow_command(self, data: list[int], expect: int, label: str,
                     timeout: float = 30.0) -> Reply | None:
        """One command that may take seconds (a load or save touches the SD
        card): sent once, waited on with a long timeout, its reply returned
        so the caller can tell BUSY from FILE_EXISTS from a real failure."""
        saved, self.timeout = self.timeout, timeout
        try:
            reply = self.send(data, expect)
        finally:
            self.timeout = saved
        if reply is not None and not reply.ok and self.verbose:
            print(f"  !! {label}: {getattr(reply.status, 'name', reply.status)}")
        return reply

    def load_song(self, name: str) -> Reply | None:
        return self.slow_command(sysex.load_song(name), sysex.Command.LOAD_SONG, f"load {name}")

    def save_song(self, name: str, overwrite: bool = False) -> Reply | None:
        return self.slow_command(sysex.save_song(name, overwrite), sysex.Command.SAVE_SONG,
                                 f"save {name}")

    # -- commands --------------------------------------------------------

    def ping(self) -> bool:
        return self.command(sysex.ping(), sysex.Command.PING, "ping")

    def select_index(self, index: int) -> bool:
        return self.command(
            sysex.select_index(index), sysex.Command.SELECT_SESSION_INDEX,
            f"select {index}",
        )

    def create(self, index, output, section=0, length=None) -> bool:
        return self.command(
            sysex.create_index(index, output, section, length),
            sysex.Command.CREATE_SESSION_INDEX, f"create {index}",
        )

    def clone(self, source, dest, section=None, length=None) -> bool:
        return self.command(
            sysex.clone_index(source, dest, section, length),
            sysex.Command.CLONE_SESSION_INDEX, f"clone {source}->{dest}",
        )

    def set_length(self, length: int) -> bool:
        return self.command(
            sysex.set_clip_length(length), sysex.Command.SET_CURRENT_CLIP_LENGTH,
            f"length {length}",
        )

    def clear(self, start=0, end=sysex.MAX_14) -> bool:
        return self.command(
            sysex.clear_clip(start, end), sysex.Command.CLEAR_CURRENT_CLIP, "clear"
        )

    def note(self, midi, position, length, velocity=100) -> bool:
        return self.command(
            sysex.set_note(midi, position, length, velocity),
            sysex.Command.SET_CURRENT_NOTE, f"note {midi}@{position}",
        )

    def name(self, text: str) -> bool:
        return self.command(
            sysex.set_clip_name(text), sysex.Command.SET_CURRENT_CLIP_NAME,
            f"name {text}",
        )

    def preset(self, text: str) -> bool:
        return self.command(
            sysex.load_synth_preset(text), sysex.Command.LOAD_CURRENT_SYNTH_PRESET,
            f"preset {text}",
        )

    def automation(self, param, start, length, start_value, end_value=None) -> bool:
        return self.command(
            sysex.set_param_automation(param, start, length, start_value, end_value),
            sysex.Command.SET_CLIP_PARAM_AUTOMATION, f"automation {param}",
        )

    def section_repeat(self, section, repeats=0, loop=False, start=None, end=None):
        return self.command(
            sysex.set_section_repeat(section, repeats, loop, start, end),
            sysex.Command.SET_SECTION_REPEAT, f"section {section}",
        )

    def clip_exists(self, index: int) -> bool:
        """Whether a session clip exists at an absolute index.

        Selecting a clip is the only probe available - there is no command
        that reports the song's clip count. Note this moves the Deluge's
        current selection as a side effect, which is harmless because every
        write path selects explicitly before it touches anything.
        """
        reply = self.send(
            sysex.select_index(index), sysex.Command.SELECT_SESSION_INDEX
        )
        return bool(reply and reply.ok)

    def count_clips(self, limit: int = 2048) -> int:
        """How many session clips the loaded song has.

        Exponential probe then binary search, so a song with 200 clips costs
        about 16 round trips rather than 200.
        """
        if not self.clip_exists(0):
            return 0
        high = 1
        while high < limit and self.clip_exists(high):
            high *= 2
        low, high = high // 2, min(high, limit)  # low exists, high does not
        while low + 1 < high:
            mid = (low + high) // 2
            if self.clip_exists(mid):
                low = mid
            else:
                high = mid
        return low + 1

    def clamp_length(self, index: int, length: int) -> dict | None:
        """Force a clip to `length` by absolute index, and read back the result.

        Unlike `set_length`, this addresses the clip by index rather than
        relying on which clip is currently selected, and the firmware replies
        with the resulting lengths - so it is both the fix and the only
        available way to verify it took.
        """
        reply = self.send(
            sysex.clamp_index_length(index, length),
            sysex.Command.CLAMP_SESSION_INDEX_LENGTH,
        )
        if reply is None or not reply.ok:
            return None
        return reply.length_detail()

    def sidechain_send(self, note, amount=1.0) -> bool:
        return self.command(
            sysex.set_sidechain_send(note, amount), sysex.Command.SET_SIDECHAIN,
            "sidechain send",
        )

    def sidechain_receive(self, amount=0.35) -> bool:
        return self.command(
            sysex.set_sidechain_receive(amount), sysex.Command.SET_SIDECHAIN,
            "sidechain receive",
        )

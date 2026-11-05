"""The Deluge's SD card over USB MIDI, no card reader needed.

The firmware answers a small JSON file protocol in SysEx (`F0 7D 04 <id>
{json} [00 packed-bytes] F7`, replies under `7D 05`). With it the Mac can list,
read and write the card while the Deluge stays plugged in, so kits, samples and
template songs get there with nobody touching the card.

Binary data travels in groups of up to eight bytes: a byte of high bits, then
up to seven bytes with their top bit cleared.
"""

from __future__ import annotations

import json
import time

CHUNK = 512
FIRST_PIECE = 30
PIECE_GAP = 0.0005


class CardError(RuntimeError):
    pass


def pack7(data: bytes) -> list[int]:
    out: list[int] = []
    for i in range(0, len(data), 7):
        group = data[i:i + 7]
        out.append(sum(((b >> 7) & 1) << n for n, b in enumerate(group)))
        out.extend(b & 0x7F for b in group)
    return out


def unpack7(data: bytes) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 8):
        high = data[i]
        out.extend(b | (((high >> n) & 1) << 7) for n, b in enumerate(data[i + 1:i + 8]))
    return bytes(out)


class Card:
    """`port` is matched by name. Port 1 stays free for riddim's own commands."""

    def __init__(self, port: str = "Port 3", timeout: float = 8.0):
        self.port = port
        self.timeout = timeout

    def __enter__(self) -> "Card":
        import rtmidi

        def find(side):
            for i, name in enumerate(side.get_ports()):
                if self.port in name:
                    return i
            raise CardError(f"no MIDI port matching {self.port!r}")

        self._in, self._out = rtmidi.MidiIn(), rtmidi.MidiOut()
        self._in.open_port(find(self._in))
        self._out.open_port(find(self._out))
        self._in.ignore_types(False, False, False)
        self._replies: list[bytes] = []
        self._in.set_callback(lambda event, _: self._replies.append(bytes(event[0])))
        time.sleep(0.2)
        # The first message on a new connection is sometimes lost.
        for attempt in range(5):
            try:
                reply, _ = self._call({"session": {"tag": "riddim"}}, msg_id=1,
                                      reply_cmd=4, timeout=1.0)
                break
            except CardError:
                if attempt == 4:
                    raise
        info = reply["^session"]
        self._ids = (info["midMin"], info["midMax"])
        self._next = info["midMin"]
        return self

    def __exit__(self, *exc) -> None:
        self._in.close_port()
        self._out.close_port()

    def _send(self, data: list[int]) -> None:
        """Send a SysEx in paced pieces. Sent whole, anything past about 40
        bytes is usually lost: the Deluge misses a message that spans USB
        transfers arriving back to back. rtmidi takes continuations of three
        bytes at most."""
        self._out.send_message(data[:FIRST_PIECE])
        for i in range(FIRST_PIECE, len(data), 3):
            time.sleep(PIECE_GAP)
            self._out.send_message(data[i:i + 3])

    def _call(self, msg: dict, payload: bytes = b"", msg_id: int | None = None,
              reply_cmd: int = 5, timeout: float | None = None) -> tuple[dict, bytes]:
        if msg_id is None:
            msg_id = self._next
            self._next = self._ids[0] if self._next >= self._ids[1] else self._next + 1
        body = list(json.dumps(msg, separators=(",", ":")).encode())
        if payload:
            body += [0] + pack7(payload)
        self._replies.clear()
        self._send([0xF0, 0x7D, 4, msg_id] + body + [0xF7])
        end = time.time() + (timeout or self.timeout)
        while time.time() < end:
            while self._replies:
                d = self._replies.pop(0)[1:-1]
                if len(d) > 3 and d[0] == 0x7D and d[1] == reply_cmd and (
                        reply_cmd == 4 or d[2] == msg_id):
                    text, _, packed = d[3:].partition(b"\0")
                    return json.loads(text), unpack7(packed)
            time.sleep(0.0005)
        raise CardError(f"no reply to {msg}")

    def list(self, path: str) -> list[dict]:
        entries: list[dict] = []
        while True:
            reply, _ = self._call({"dir": {"path": path, "offset": len(entries), "lines": 20}})
            page = reply["^dir"].get("list", [])
            if reply["^dir"].get("err"):
                raise CardError(f"dir {path}: error {reply['^dir']['err']}")
            entries += page
            if len(page) < 20:
                return entries

    def read(self, path: str) -> bytes:
        reply, _ = self._call({"open": {"path": path, "write": 0}})
        info = reply["^open"]
        if info.get("err"):
            raise CardError(f"open {path}: error {info['err']}")
        fid, size, data = info["fid"], info["size"], bytearray()
        try:
            while len(data) < size:
                head, chunk = self._call(
                    {"read": {"fid": fid, "addr": len(data), "size": CHUNK}})
                if not chunk:
                    raise CardError(f"read {path} stopped at {len(data)}: {head}")
                data += chunk
        finally:
            self._call({"close": {"fid": fid}})
        return bytes(data[:size])

    def write(self, path: str, data: bytes) -> None:
        """Write a file, then read it back: a card holding a half-written song
        fails at the worst time, so a mismatch is an error here."""
        reply, _ = self._call({"open": {"path": path, "write": 1}})
        info = reply["^open"]
        if info.get("err"):
            raise CardError(f"open {path} for write: error {info['err']}")
        fid = info["fid"]
        try:
            for addr in range(0, len(data), CHUNK):
                part = data[addr:addr + CHUNK]
                head, _ = self._call(
                    {"write": {"fid": fid, "addr": addr, "size": len(part)}}, part)
                if head.get("^write", {}).get("err"):
                    raise CardError(f"write {path} at {addr}: {head}")
        finally:
            self._call({"close": {"fid": fid}})
        if self.read(path) != data:
            raise CardError(f"{path} did not read back the same")

    def mkdir(self, path: str) -> None:
        self._call({"mkdir": {"path": path}})

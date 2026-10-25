"""Collect reference recordings to listen to: run on your own machine.

A list of URLs in (one per line, `#` comments, an optional tag after the
URL), audio and its provenance out. A URL is anything yt-dlp reads, or an
archive.org item (archive.org/details/<item>), whose audio files are all
fetched straight from the archive with its metadata:

    references/
      audio/<id>.m4a          the recording
      audio/<id>.info.json    everything the source says about it
      manifest.jsonl          one line per recording: id, url, title,
                              uploader, duration, tags, fetched-at,
                              and the riddim listening report once
                              `riddim-cli listen --manifest` has run

    python ml/collect_refs.py refs.txt                 # into ./references
    python ml/collect_refs.py refs.txt --out ~/refs
    python ml/collect_refs.py refs.txt --prefer-mp3   # the archive's MP3s, not FLAC/WAV

refs.txt:
    https://www.youtube.com/watch?v=7W-qIw4nnow   dubstep dmz
    https://...                                   roots_dub

Already-fetched ids are skipped, so the list can grow and be re-run. Needs
yt-dlp and ffmpeg (`brew install yt-dlp ffmpeg`). For personal, non-
commercial study: the manifest keeps where every file came from, so it is
always possible to say what any later model was trained on.
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

AUDIO_EXT = (".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aiff", ".aif")
PREFER_MP3 = "--prefer-mp3" in sys.argv


def _get(url, path=None, tries=4):
    """Fetch with retries: the archive throws the odd 500 under load."""
    for k in range(tries):
        try:
            return _get_once(url, path)
        except Exception as e:      # noqa: BLE001 - network, HTTP, partial reads
            if k == tries - 1:
                raise
            print(f"  retry {k + 1} ({e}) {url}", flush=True)
            time.sleep(5 * (k + 1))


def _get_once(url, path=None):
    req = urllib.request.Request(url, headers={"User-Agent": "riddim-collect/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if path is None:
            return r.read()
        with open(path + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    os.replace(path + ".part", path)


def audible(path):
    """False for a file that decodes to digital silence (some uploads are
    empty transfers) - None when ffmpeg is not there to ask."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-t", "90", "-i", path, "-af",
                              "volumedetect", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=120).stderr
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        if "max_volume:" in line:
            return float(line.split("max_volume:")[1].split()[0]) > -60.0
    return None


def archive_item(url, tags, audio, seen, write):
    """Every audio file of an archive.org item, with the item's metadata.
    Items without a licence are marked so: fine to study, not clean data."""
    item = urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1]
    meta = json.loads(_get(f"https://archive.org/metadata/{item}"))
    m = meta.get("metadata", {})
    files = [f for f in meta.get("files", []) if f["name"].lower().endswith(AUDIO_EXT)
             # the archive's low-bitrate and preview copies of the same tracks
             and not f["name"].lower().endswith(("_64kb.mp3", "_vbr.mp3", "_sample.mp3"))]
    # The archive derives MP3s from what was uploaded: take the originals
    # only, so a FLAC album is not fetched twice.
    originals = [f for f in files if f.get("source") != "derivative"]
    if PREFER_MP3:
        # Lossless is wasted on listening: take the archive's MP3s where
        # it made them, the originals only where it did not.
        mp3s = [f for f in files if f["name"].lower().endswith(".mp3")]
        stems = {os.path.splitext(f["name"])[0] for f in mp3s}
        originals = mp3s + [f for f in originals if os.path.splitext(f["name"])[0] not in stems]
    for f in originals or files:
        name = f["name"]
        rid = "ia_" + hashlib.sha1(f"{item}/{name}".encode()).hexdigest()[:12]
        with MANIFEST_LOCK:
            if rid in seen:
                continue
            seen.add(rid)
        ext = os.path.splitext(name)[1].lower()
        dest = os.path.join(audio, rid + ext)
        try:
            _get(f"https://archive.org/download/{item}/{urllib.parse.quote(name)}", dest)
        except Exception as e:      # noqa: BLE001
            print(f"failed {item}/{name}: {e} (re-run to retry)", flush=True)
            with MANIFEST_LOCK:
                seen.discard(rid)
            continue
        folder, base = os.path.split(name)
        write({"id": rid, "url": f"https://archive.org/details/{item}", "source": "archive.org",
               "item": item, "path_in_item": name,
               "title": os.path.splitext(base)[0], "release": folder or m.get("title"),
               "artist": m.get("creator"), "uploader": m.get("uploader"),
               "license": m.get("licenseurl") or "none stated (unofficial upload - study only)",
               "tags": tags, "file": os.path.join("audio", rid + ext),
               "bytes": int(f.get("size", 0)), "md5": f.get("md5"),
               "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "silent": audible(dest) is False})
        print(f"got {rid}  {name}", flush=True)
        time.sleep(1)


YTDLP = None
MANIFEST_LOCK = threading.Lock()
BOT_CHECK = ("confirm you", "not a bot", "sign in to confirm")


def _ytdlp():
    """yt-dlp as a module of this python when it is installed there (the
    project venv), else the command on PATH."""
    global YTDLP
    if YTDLP is None:
        try:
            import yt_dlp  # noqa: F401
            YTDLP = [sys.executable, "-m", "yt_dlp"]
        except ImportError:
            YTDLP = ["yt-dlp"]
    return YTDLP


def _have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=20)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def _entries(url, cap):
    """The single videos/tracks/shows behind a URL: a playlist, channel,
    user page or album expands to its entries (newest first, at most `cap`),
    a single item is itself. Returns (entries, error)."""
    r = subprocess.run(_ytdlp() + ["--flat-playlist", "--playlist-end", str(cap),
                                   "--print", "%(id)s\t%(webpage_url,url)s", url],
                       capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        rid, _, u = line.partition("\t")
        if rid and rid != "NA":
            out.append((rid, u if u and u != "NA" else url))
    return out, r.stderr


def stream_source(url, tags, audio, seen, write, ffmpeg):
    """Everything yt-dlp reads: expand, then fetch each entry as audio.
    A site asking to prove we are not a bot is respected: the source is
    skipped, never worked around."""
    cap = 40
    for t in tags:
        if t.startswith("max:"):
            cap = int(t[4:])
    tags = [t for t in tags if not t.startswith("max:")]
    entries, err = _entries(url, cap)
    if not entries:
        print(f"skip (nothing listed): {url}\n  {err.strip()[-300:]}", flush=True)
        return
    host = urllib.parse.urlparse(url).netloc.replace("www.", "").split(".")[0]
    for rid0, u in entries:
        rid = f"{host}_{rid0}"
        with MANIFEST_LOCK:
            if rid in seen:
                continue
            seen.add(rid)
        tmpl = os.path.join(audio, rid + ".%(ext)s")
        fmt = ["-x", "--audio-format", "m4a"] if ffmpeg else ["-f", "bestaudio[ext=m4a]/bestaudio"]
        r = subprocess.run(_ytdlp() + ["--no-playlist", *fmt, "--write-info-json",
                                       "--no-progress", "-o", tmpl, u],
                           capture_output=True, text=True)
        meta_path = os.path.join(audio, rid + ".info.json")
        if any(b in r.stderr.lower() for b in BOT_CHECK):
            print(f"bot check at {host} - leaving {url} alone", flush=True)
            with MANIFEST_LOCK:
                seen.discard(rid)
            return
        if r.returncode != 0 or not os.path.exists(meta_path):
            print(f"failed {u}\n  {r.stderr.strip()[-300:]}", flush=True)
            with MANIFEST_LOCK:
                seen.discard(rid)
            continue
        meta = json.load(open(meta_path))
        got = [f for f in os.listdir(audio) if f.startswith(rid + ".") and not f.endswith(".json")]
        write({"id": rid, "url": u, "source": host, "from": url, "title": meta.get("title"),
               "uploader": meta.get("uploader"), "duration": meta.get("duration"),
               "upload_date": meta.get("upload_date"),
               "license": meta.get("license") or "none stated - study only",
               "tags": tags, "file": os.path.join("audio", got[0]) if got else None,
               "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"got {rid}: {meta.get('uploader')} - {meta.get('title')}", flush=True)
        time.sleep(2)     # be polite


def main():
    """collect_refs.py LIST [LIST...] [--out DIR] [--prefer-mp3] [--jobs N]

    Sources run N at a time (default 4); a line's `max:N` tag caps how many
    entries a playlist / channel / user page contributes (default 40)."""
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    out = os.path.expanduser(args[args.index("--out") + 1]) if "--out" in args else "references"
    jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 4
    skip = {args[i + 1] for i, a in enumerate(args) if a in ("--out", "--jobs")}
    srcs = [a for a in args if not a.startswith("--") and a not in skip]
    audio = os.path.join(out, "audio")
    os.makedirs(audio, exist_ok=True)
    manifest = os.path.join(out, "manifest.jsonl")
    seen = set()
    if os.path.exists(manifest):
        for line in open(manifest):
            try:
                seen.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                pass

    def write(row):
        with MANIFEST_LOCK:
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")

    lines = []
    for src in srcs:
        for line in open(src):
            line = line.split("#", 1)[0].strip()
            if line:
                url, *tags = line.split()
                lines.append((url, tags))
    ffmpeg = _have_ffmpeg()
    print(f"{len(lines)} sources from {len(srcs)} lists, {jobs} at a time, "
          f"ffmpeg {'yes' if ffmpeg else 'no (native audio formats)'}", flush=True)

    def one(job):
        url, tags = job
        try:
            if "archive.org/details/" in url:
                archive_item(url, tags, audio, seen, write)
            else:
                stream_source(url, tags, audio, seen, write, ffmpeg)
        except Exception as e:  # noqa: BLE001
            print(f"failed {url}: {e} (re-run to retry)", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, _ in enumerate(pool.map(one, lines), 1):
            print(f"[{i}/{len(lines)}] sources done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

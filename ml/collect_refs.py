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
import time
import urllib.parse
import urllib.request

AUDIO_EXT = (".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aiff", ".aif")


def _get(url, path=None):
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


def archive_item(url, tags, audio, seen, write):
    """Every audio file of an archive.org item, with the item's metadata.
    Items without a licence are marked so: fine to study, not clean data."""
    item = urllib.parse.urlparse(url).path.rstrip("/").split("/")[-1]
    meta = json.loads(_get(f"https://archive.org/metadata/{item}"))
    m = meta.get("metadata", {})
    for f in meta.get("files", []):
        name = f["name"]
        if not name.lower().endswith(AUDIO_EXT):
            continue
        rid = "ia_" + hashlib.sha1(f"{item}/{name}".encode()).hexdigest()[:12]
        if rid in seen:
            print(f"have {rid}  {name}")
            continue
        ext = os.path.splitext(name)[1].lower()
        dest = os.path.join(audio, rid + ext)
        _get(f"https://archive.org/download/{item}/{urllib.parse.quote(name)}", dest)
        folder, base = os.path.split(name)
        write({"id": rid, "url": f"https://archive.org/details/{item}", "source": "archive.org",
               "item": item, "path_in_item": name,
               "title": os.path.splitext(base)[0], "release": folder or m.get("title"),
               "artist": m.get("creator"), "uploader": m.get("uploader"),
               "license": m.get("licenseurl") or "none stated (unofficial upload - study only)",
               "tags": tags, "file": os.path.join("audio", rid + ext),
               "bytes": int(f.get("size", 0)), "md5": f.get("md5"),
               "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")})
        seen.add(rid)
        print(f"got {rid}  {name}")
        time.sleep(1)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    out = os.path.expanduser(args[args.index("--out") + 1]) if "--out" in args else "references"
    src = args[0]
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

    for line in open(src):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        url, *tags = line.split()
        if "archive.org/details/" in url:
            def write(row):
                with open(manifest, "a") as f:
                    f.write(json.dumps(row) + "\n")
            archive_item(url, tags, audio, seen, write)
            continue
        info = subprocess.run(["yt-dlp", "--no-playlist", "--print", "id", url],
                              capture_output=True, text=True)
        rid = info.stdout.strip()
        if not rid:
            print(f"skip (no id): {url}\n  {info.stderr.strip()[-300:]}")
            continue
        if rid in seen:
            print(f"have {rid}")
            continue
        r = subprocess.run(["yt-dlp", "--no-playlist", "-x", "--audio-format", "m4a",
                            "--write-info-json", "-o", os.path.join(audio, "%(id)s.%(ext)s"),
                            url], capture_output=True, text=True)
        meta_path = os.path.join(audio, f"{rid}.info.json")
        if r.returncode != 0 or not os.path.exists(meta_path):
            print(f"failed {url}\n  {r.stderr.strip()[-300:]}")
            continue
        meta = json.load(open(meta_path))
        row = {"id": rid, "url": url, "title": meta.get("title"),
               "uploader": meta.get("uploader"), "duration": meta.get("duration"),
               "upload_date": meta.get("upload_date"), "license": meta.get("license"),
               "tags": tags, "file": os.path.join("audio", f"{rid}.m4a"),
               "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(manifest, "a") as f:
            f.write(json.dumps(row) + "\n")
        seen.add(rid)
        print(f"got {rid}: {row['uploader']} - {row['title']}")
        time.sleep(2)     # be polite
    return 0


if __name__ == "__main__":
    sys.exit(main())

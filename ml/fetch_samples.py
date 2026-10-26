"""Fetch free sample packs listed in samples_direct.tsv into samples/packs/.

    python ml/fetch_samples.py samples_direct.tsv [--out samples/packs]

Plain GETs only: rows flagged CHECK (sites that refuse a plain client) are
skipped, never worked around. Zips are unpacked beside themselves (rar packs
need unar/7z); every pack's licence goes into LICENCES.tsv next to them.
Already-fetched packs are skipped, so it can be re-run.
"""

import csv
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile


def main():
    args = sys.argv[1:]
    src = args[0]
    out = args[args.index("--out") + 1] if "--out" in args else "samples/packs"
    os.makedirs(out, exist_ok=True)
    lic_path = os.path.join(out, "LICENCES.tsv")
    have = set()
    if os.path.exists(lic_path):
        have = {line.split("\t", 1)[0] for line in open(lic_path)}
    rows = list(csv.DictReader(open(src), delimiter="\t"))
    for r in rows:
        name = r["dest_filename"]
        if "CHECK" in (r.get("contents_note") or "") or name in have:
            continue
        dest = os.path.join(out, name)
        try:
            req = urllib.request.Request(r["url"], headers={"User-Agent": "riddim-samples/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(resp, f, 1 << 20)
            os.replace(dest + ".part", dest)
        except Exception as e:  # noqa: BLE001
            print(f"failed {name}: {e}", flush=True)
            continue
        folder = os.path.join(out, os.path.splitext(name)[0])
        try:
            if zipfile.is_zipfile(dest):
                with zipfile.ZipFile(dest) as z:
                    z.extractall(folder)
                os.remove(dest)
            elif name.lower().endswith(".rar"):
                for tool in (["unar", "-q", "-o", folder], ["7z", "x", f"-o{folder}", "-y"]):
                    try:
                        if subprocess.run(tool + [dest], capture_output=True).returncode == 0:
                            os.remove(dest)
                            break
                    except OSError:
                        continue
        except Exception as e:  # noqa: BLE001
            print(f"unpack failed {name}: {e} (kept the archive)", flush=True)
        with open(lic_path, "a") as f:
            f.write("\t".join([name, r["pack_name"], r["licence"], r["url"]]) + "\n")
        print(f"got {name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download one large file as many parallel byte ranges (resumable).

usage: rangefetch.py URL OUT WORKERS SIZE_BYTES
"""
import os, sys, concurrent.futures as cf, urllib.request, time

url, out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
size = int(sys.argv[4])
chunk = 8 * 1024 * 1024
parts = os.path.join(os.path.dirname(out), ".parts"); os.makedirs(parts, exist_ok=True)

def fetch(i):
    a = i * chunk; b = min(size, a + chunk) - 1
    p = os.path.join(parts, f"{i:05d}")
    if os.path.exists(p) and os.path.getsize(p) == b - a + 1:
        return 0
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            if len(data) == b - a + 1:
                with open(p + ".tmp", "wb") as f: f.write(data)
                os.replace(p + ".tmp", p)
                return len(data)
        except Exception:
            time.sleep(2 + attempt * 3)
    raise RuntimeError(f"chunk {i} failed")

n = (size + chunk - 1) // chunk
done = 0; t = time.time()
with cf.ThreadPoolExecutor(workers) as ex:
    for k, got in enumerate(ex.map(fetch, range(n)), 1):
        done += got
        if k % 10 == 0 or k == n:
            print(f"{k}/{n} chunks, {done/1e6:.0f} MB this run, {done/1e3/max(1,time.time()-t):.0f} KB/s", flush=True)
with open(out, "wb") as o:
    for i in range(n):
        with open(os.path.join(parts, f"{i:05d}"), "rb") as f: o.write(f.read())
print("ASSEMBLED", os.path.getsize(out))

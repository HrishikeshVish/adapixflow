"""Stream a prefix of a remote .tar.gz (official Kubric-4D release) and extract only the members of the named scenes.

Untrusted archive: members are extracted with tarfile's 'data' filter (no absolute paths, links outside, devices); nothing
in the archive is executed. Scenes not in --scenes are read past (not written). The stream stops when every wanted scene
is complete (a member of another scene follows it; the archive stores a scene's members contiguously, see members.tsv),
or after --max-gb compressed bytes.
Resume safety:
  - a dropped connection is resumed inside the run with an HTTP Range request from the current compressed offset (the
    gzip state stays in this process, so no byte is decoded twice);
  - a completed scene gets <out>/done_<scene>.json (member count, bytes, seconds); a rerun skips writing scenes that have
    it, and re-extracts (overwrites) a scene that has none.
Writes <out>/members.tsv (name, size, kept) and <out>/stream.json.
Usage (python -I): stream_prefix.py <url> <out_dir> --scenes scn02801,scn02802 [--max-gb 16]
"""

from __future__ import annotations

import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


class Resuming:
    """File-like over an HTTP body that reconnects with a Range header on a read error (up to `tries` times)."""

    def __init__(self, url, tries=20):
        self.url, self.n, self.tries, self.resumes = url, 0, tries, 0
        self.f = urllib.request.urlopen(url, timeout=120)

    def read(self, k=-1):
        for attempt in range(self.tries):
            try:
                b = self.f.read(k)
                self.n += len(b)
                return b
            except Exception as e:  # noqa: BLE001  (network error: reconnect at the current offset)
                print(f"read error at {self.n} B ({e!r}); resuming", flush=True)
                time.sleep(min(60, 5 * (attempt + 1)))
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.n}-"})
                try:
                    self.f = urllib.request.urlopen(req, timeout=120)
                    assert self.f.status == 206, f"server ignored Range (status {self.f.status})"
                    self.resumes += 1
                except Exception as e2:  # noqa: BLE001
                    print(f"reconnect failed ({e2!r})", flush=True)
        raise IOError("too many read errors")


def scene_of(name):
    return next((p for p in Path(name).parts if p.startswith("scn")), None)


def main():
    url, out = sys.argv[1], Path(sys.argv[2])
    wanted = sys.argv[sys.argv.index("--scenes") + 1].split(",")
    max_b = float(sys.argv[sys.argv.index("--max-gb") + 1]) * 1e9 if "--max-gb" in sys.argv else 16e9
    out.mkdir(parents=True, exist_ok=True)
    done = {s for s in wanted if (out / f"done_{s}.json").exists()}
    resp, t0 = Resuming(url), time.time()
    order, cur, stat = [], None, {}
    stop = "end of archive"
    with tarfile.open(fileobj=resp, mode="r|gz") as tf, open(out / "members.tsv", "w") as log:
        for m in tf:
            sc = scene_of(m.name)
            if sc != cur:  # scene boundary: the previous scene is complete
                if cur in stat and cur in wanted and cur not in done:
                    st = stat[cur]
                    st["seconds"] = round(time.time() - st.pop("t0"), 1)
                    st["compressed_offset_end"] = resp.n
                    (out / f"done_{cur}.json").write_text(json.dumps({"scene": cur, **st}))
                    done.add(cur)
                    print(f"done {cur}: {st}", flush=True)
                if sc is not None and sc in order:
                    stop = f"scene {sc} not contiguous"
                    log.write(f"{m.name}\t{m.size}\tSTOP({stop})\n")
                    break
                if sc is not None:
                    order.append(sc)
                cur = sc
                if all(s in done for s in wanted):
                    stop = "all wanted scenes complete"
                    log.write(f"{m.name}\t{m.size}\tSTOP({stop})\n")
                    break
                if sc is not None:
                    stat[sc] = {"members": 0, "bytes": 0, "t0": time.time(), "compressed_offset_start": resp.n}
            keep = sc in wanted and sc not in done
            if sc is not None:
                stat[sc]["members"] += 1
                stat[sc]["bytes"] += m.size
            if keep:
                tf.extract(m, out / "extract", filter="data")
            log.write(f"{m.name}\t{m.size}\t{int(keep)}\n")
            if resp.n > max_b:
                stop = f"max bytes {resp.n}"
                log.write(f"STOP({stop})\n")
                break
    rep = {"url": url, "wanted": wanted, "done": sorted(done), "archive_scene_order": order, "stop": stop,
           "compressed_read_gb": round(resp.n / 1e9, 3), "resumes": resp.resumes, "seconds": round(time.time() - t0)}
    (out / "stream.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()

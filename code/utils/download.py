"""MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Changes: downloads use HTTP range requests that resume after connection drops
(Zenodo regularly cuts long transfers). With --only, just the byte spans of the
selected archive members (plus the zip's central directory) are fetched into a
sparse file, so e.g. the 626MB test set can be extracted without downloading or
storing the full 1.6GB data.zip. Already extracted files are skipped on re-runs.
"""

import argparse
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://zenodo.org/records/3572653/files/"
DEFAULT_FILES = ["weights.zip", "data.zip"]
CHUNK = 1 << 20


def remote_size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


def fetch_range(url, fh, start, end, retries=20):
    """Write bytes [start, end) of url into fh at the same offset, resuming on errors."""
    pos, attempt = start, 0
    while pos < end:
        req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{end - 1}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError(f"server ignored range request (HTTP {r.status})")
                fh.seek(pos)
                while pos < end:
                    chunk = r.read(min(CHUNK, end - pos))
                    if not chunk:
                        break
                    fh.write(chunk)
                    pos += len(chunk)
                    attempt = 0
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            attempt += 1
            if attempt > retries:
                raise
            wait = 30 if getattr(e, "code", None) == 429 else min(2 * attempt, 30)
            print(f"  connection problem at byte {pos} ({e}); retry {attempt} in {wait}s", flush=True)
            time.sleep(wait)


def spans_for(infos, size, gap=8 * CHUNK):
    """Merge member byte ranges (local header + data) into few contiguous spans."""
    ranges = sorted((i.header_offset, min(size, i.header_offset + 30 + len(i.filename) + 1024 + i.compress_size))
                    for i in infos)
    spans = []
    for a, b in ranges:
        if spans and a - spans[-1][1] <= gap:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return spans


def download_extract(url, dest, item, only=None):
    size = remote_size(url)
    partial = dest / f".partial_{item}"
    partial.touch()
    with open(partial, "r+b") as fh:  # not append mode: writes must land at their seek offset
        fh.truncate(size)  # sparse: only the fetched spans take disk space
        # 1) central directory at the end of the archive (grow the tail until zipfile can parse it)
        tail = min(size, 4 * CHUNK)
        while True:
            fetch_range(url, fh, size - tail, size)
            fh.flush()
            try:
                infos = zipfile.ZipFile(partial).infolist()
                break
            except zipfile.BadZipFile:
                if tail == size:
                    raise
                tail = min(size, tail * 4)

        infos = [i for i in infos if not i.is_dir()]
        if only:
            infos = [i for i in infos if any(o in i.filename for o in only)]
        todo = [i for i in infos
                if not ((dest / i.filename).exists() and (dest / i.filename).stat().st_size == i.file_size)]
        print(f"  {len(infos)} files selected, {len(todo)} to fetch "
              f"({sum(i.compress_size for i in todo) / 1e6:.0f}MB)", flush=True)

        # 2) the selected members' bytes
        for n, (a, b) in enumerate(spans_for(todo, size), 1):
            print(f"  span {n}: {(b - a) / 1e6:.0f}MB", flush=True)
            fetch_range(url, fh, a, b)
        fh.flush()

    # 3) extract with the standard library (CRC-checked)
    with zipfile.ZipFile(partial) as z:
        for n, info in enumerate(todo, 1):
            z.extract(info, dest)
            if n % 1000 == 0:
                print(f"  extracted {n}/{len(todo)}", flush=True)
    partial.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Pollen Files")
    parser.add_argument("-p", "--path", type=str, default="../",
                        help="The path to where the data should be downloaded")
    parser.add_argument("-f", "--files", type=str, nargs="+", default=DEFAULT_FILES,
                        help="The files to be downloaded")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="only extract archive members whose path contains one of these strings, "
                             "e.g. --only pollen_20190523 (test set, 626MB) to save disk space")
    args = parser.parse_args()

    data_dir = Path(args.path)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading files to {data_dir}")
    for item in args.files:
        print(f"Fetching {item}", flush=True)
        download_extract(BASE_URL + item + "?download=1", data_dir, item, only=args.only)
    print("Done")

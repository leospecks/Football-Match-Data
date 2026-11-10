"""Download every SkillCorner match folder into this project's data folder.

Run: python fetch_data.py
Optional: python fetch_data.py --data-dir data --overwrite
Uses only Python's standard library. No Git or extra packages required.
Existing files with the expected size are skipped unless --overwrite is set.
Downloads use temporary files and checksum verification before atomic rename.
Interrupted downloads restart from the beginning on the next run.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from http.client import IncompleteRead
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPOSITORY = "SkillCorner/opendata"
BRANCH = "master"
SOURCE_DIR = "data/matches"
DATA_DIR = Path(__file__).resolve().parent / "data"
RETRIES = 3
TIMEOUT = 90
CHUNK_SIZE = 1024 * 1024
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def _open(url):
    headers = {"User-Agent": "SkillCorner-match-downloader"}
    if url.startswith("https://api.github.com/"):
        headers["Accept"] = "application/vnd.github+json"
    return urlopen(Request(url, headers=headers), timeout=TIMEOUT)


def _retry(action, label):
    for attempt in range(RETRIES + 1):
        try:
            return action()
        except HTTPError as error:
            retryable = error.code in (429, 500, 502, 503, 504)
            if error.code == 403:
                error.close()
                raise RuntimeError(
                    f"GitHub denied {label}; its API/download rate limit may be reached. "
                    "Try again later. Completed files will be skipped."
                ) from error
            if not retryable or attempt == RETRIES:
                error.close()
                raise
            retry_after = error.headers.get("Retry-After", "")
            delay = min(30, int(retry_after)) if retry_after.isdigit() else 2 ** attempt
            error.close()
        except (URLError, TimeoutError, ConnectionError, IncompleteRead, ValueError) as error:
            if attempt == RETRIES:
                raise
            delay = 2 ** attempt
        print(f"  Retrying {label} in {delay}s ({attempt + 1}/{RETRIES})...")
        time.sleep(delay)


def _json(url):
    def read():
        with _open(url) as response:
            return json.load(response)
    return _retry(read, "repository listing")


def _tree(sha, recursive=False):
    suffix = "?recursive=1" if recursive else ""
    return _json(f"{API_ROOT}/git/trees/{quote(sha, safe='')}{suffix}")


def discover_files():
    """Return an immutable commit ID and every regular file in match folders."""
    commit = _json(f"{API_ROOT}/commits/{quote(BRANCH, safe='')}")
    commit_sha = commit["sha"]
    root_sha = commit["commit"]["tree"]["sha"]
    listing = _tree(root_sha, recursive=True)
    if not listing.get("truncated"):
        entries = listing["tree"]
    else:
        # GitHub can truncate large recursive listings: walk relevant subtrees
        # instead of silently omitting some match files.
        entries, pending = [], [(root_sha, "")]
        while pending:
            sha, prefix = pending.pop()
            subtree = _tree(sha)
            if subtree.get("truncated"):
                raise RuntimeError("GitHub truncated a directory listing; no complete download is possible.")
            for entry in subtree["tree"]:
                path = f"{prefix}/{entry['path']}".lstrip("/")
                relevant = (path == SOURCE_DIR or SOURCE_DIR.startswith(path + "/")
                            or path.startswith(SOURCE_DIR + "/"))
                if not relevant:
                    continue
                if entry["type"] == "tree":
                    pending.append((entry["sha"], path))
                else:
                    entries.append({**entry, "path": path})
    files = []
    for entry in entries:
        path = entry["path"]
        if not path.startswith(SOURCE_DIR + "/") or entry["type"] != "blob":
            continue
        relative = PurePosixPath(path).relative_to(SOURCE_DIR)
        if len(relative.parts) < 2:
            continue  # Only files inside match folders, not sibling metadata.
        if entry.get("mode") not in (None, "100644", "100755"):
            raise RuntimeError(f"Unsupported non-regular repository file: {path}")
        files.append(entry)
    if not files:
        raise RuntimeError(f"No match files found under {SOURCE_DIR}.")
    return commit_sha, sorted(files, key=lambda entry: entry["path"])


def _target(path, data_dir):
    relative = PurePosixPath(path).relative_to(SOURCE_DIR)
    if any(part in (".", "..") or "\\" in part or ":" in part for part in relative.parts):
        raise ValueError(f"Invalid repository path: {path}")
    target = data_dir.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError(f"Destination escapes the data folder: {path}")
    return target


def _lfs_pointer(head):
    if not head.startswith(LFS_HEADER):
        return None
    fields = dict(line.split(" ", 1) for line in head.decode("utf-8").splitlines() if " " in line)
    oid = fields.get("oid", "")
    digest = oid.removeprefix("sha256:")
    if not oid.startswith("sha256:") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid Git LFS pointer checksum.")
    size = int(fields["size"])
    if size < 0:
        raise ValueError("Invalid Git LFS file size.")
    return size, digest


def _write_file(response, head, target, size, expected_digest, algorithm):
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.new(algorithm)
    if algorithm == "sha1":
        hasher.update(f"blob {size}\0".encode())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".",
                                         suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            written = 0
            block = head or response.read(CHUNK_SIZE)
            while block:
                output.write(block)
                hasher.update(block)
                written += len(block)
                block = response.read(CHUNK_SIZE)
        if written != size or hasher.hexdigest() != expected_digest:
            raise ValueError(f"Size/checksum verification failed for {target.name}.")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return size


def download_file(entry, commit_sha, data_dir, overwrite=False):
    """Download the real content, resolving Git LFS pointers when necessary."""
    target = _target(entry["path"], data_dir)
    remote_path = quote(entry["path"], safe="/")
    raw_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit_sha}/{remote_path}"

    def download():
        with _open(raw_url) as response:
            head = response.read(1024)
            pointer = _lfs_pointer(head)
            size = pointer[0] if pointer else entry["size"]
            if not overwrite and target.is_file() and target.stat().st_size == size:
                return "skipped", size
            print(f"  Downloading {size / 1024 / 1024:.1f} MiB...", flush=True)
            if pointer:
                response.close()
                media_url = f"https://media.githubusercontent.com/media/{REPOSITORY}/{commit_sha}/{remote_path}"
                with _open(media_url) as content:
                    _write_file(content, b"", target, size, pointer[1], "sha256")
            else:
                _write_file(response, head, target, size, entry["sha"], "sha1")
        return "downloaded", size

    return _retry(download, str(target.relative_to(data_dir)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Destination folder (default: data next to this script).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download files even when an existing file has the expected size.")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.resolve()
    commit, files = discover_files()
    match_ids = {PurePosixPath(e["path"]).relative_to(SOURCE_DIR).parts[0] for e in files}
    print(f"Found {len(files)} files across {len(match_ids)} match folders.")
    print(f"Saving to {data_dir}")
    downloaded = skipped = failed = 0
    for index, entry in enumerate(files, 1):
        relative = PurePosixPath(entry["path"]).relative_to(SOURCE_DIR)
        print(f"[{index}/{len(files)}] {relative}", flush=True)
        try:
            status, size = download_file(entry, commit, data_dir, args.overwrite)
            downloaded += status == "downloaded"
            skipped += status == "skipped"
            print(f"  {status.capitalize()} ({size / 1024 / 1024:.1f} MiB)", flush=True)
        except (OSError, ValueError, RuntimeError, IncompleteRead) as error:
            failed += 1
            print(f"  FAILED: {error}", file=sys.stderr)
    print(f"Done: {downloaded} downloaded, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped. Run again to download the remaining files.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, IncompleteRead) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)

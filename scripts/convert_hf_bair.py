"""Stream BAIR tar members into Parquet without extracting files or decoding video.

The repackaged MP4 bytes are preserved exactly; action arrays retain float64 precision.
Use notebook 01's SOURCE='parquet' route to decode them for the existing experiments.
"""
from pathlib import Path, PurePosixPath
import argparse
import io
import json
import subprocess
import tarfile
from contextlib import contextmanager

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "guyuchao/BAIR"
REVISION = "776ad3b"
SCHEMA = pa.schema([
    ("clip_id", pa.string()), ("source_split", pa.string()),
    ("video_aux1_mp4", pa.binary()), ("actions", pa.list_(pa.list_(pa.float64(), 4))),
    ("source_shard", pa.string()),
])


@contextmanager
def open_http(url):
    # System curl honors the host's certificate configuration. Some local
    # Anaconda installs have a separate CA bundle and reject valid proxy chains.
    process = subprocess.Popen(
        ["curl", "--location", "--fail", "--silent", "--show-error",
         "--connect-timeout", "30", url], stdout=subprocess.PIPE)
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()  # expected when a clip quota closes the stream early
        process.wait(timeout=30)


def samples(stream, wanted, source, allow_early_close=False):
    """Group paired camera/action members; never extract untrusted tar paths."""
    pending = {}
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if len(path.parts) < 4:
                continue
            _, split, kind, filename = path.parts[-4:]
            if split not in wanted or wanted[split] <= 0:
                continue
            if kind not in {"video_aux1", "action_pos"}:
                continue
            if member.size > 20_000_000:
                raise ValueError(f"Unexpectedly large member: {member.name}")
            key = f"{split}/{Path(filename).stem}"
            item = pending.setdefault(key, {})
            if kind in item:
                raise ValueError(f"Repeated member: {member.name}")
            item[kind] = archive.extractfile(member).read()
            if "video_aux1" in item and "action_pos" in item:
                with np.load(io.BytesIO(item["action_pos"]), allow_pickle=False) as arrays:
                    actions = np.asarray(arrays["action"], dtype=np.float64)
                if actions.shape != (30, 4) or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid actions: {key}")
                yield dict(clip_id=key, source_split=split,
                           video_aux1_mp4=item["video_aux1"], actions=actions.tolist(),
                           source_shard=source)
                del pending[key]
                wanted[split] -= 1
                # Early-close a shard once the requested quota for its split is met.
                # A later selected shard can still supply the other split.
                if all(n == 0 for n in wanted.values()) or (allow_early_close and wanted[split] == 0):
                    return
            if len(pending) > 10000:
                raise ValueError("Too many unpaired members; check shard ordering/schema.")


def convert(output, train_clips=1000, test_clips=256, shards=(0, 4), local_paths=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = output / "bair_subset.parquet"
    if final.exists():
        raise FileExistsError(f"Preserving existing output: {final}")
    temporary = output / "bair_subset.parquet.partial"
    if temporary.exists():
        raise FileExistsError(f"Prior interrupted output exists: {temporary}; use another directory.")
    wanted = {"train": train_clips, "test": test_clips}
    if train_clips < 2 or test_clips < 1:
        raise ValueError("Request at least 2 training clips and 1 test clip.")
    sources = list(local_paths) if local_paths else [
        f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/shard-{i:05d}.tar"
        for i in shards]
    counts, seen = {"train": 0, "test": 0}, set()
    with pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:
        buffer = []
        for source_index, source in enumerate(sources):
            print("Reading:", source, flush=True)
            if str(source).startswith("https://"):
                stream = open_http(source)
            else:
                stream = open(source, "rb")
            with stream as opened:
                for row in samples(opened, wanted, str(source),
                                   allow_early_close=source_index < len(sources) - 1):
                    if row["clip_id"] in seen:
                        raise ValueError(f"Duplicate source clip: {row['clip_id']}")
                    seen.add(row["clip_id"])
                    counts[row["source_split"]] += 1
                    buffer.append(row)
                    if len(buffer) >= 64:
                        writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
                        buffer = []
            print("Converted:", counts, flush=True)
            if all(n == 0 for n in wanted.values()):
                break
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
    if any(n > 0 for n in wanted.values()):
        raise RuntimeError(f"Selected shards did not supply the requested clips: missing {wanted}. "
                           f"Partial output preserved at {temporary}. Add shards/use a new output directory.")
    temporary.rename(final)
    (output / "source.json").write_text(json.dumps(dict(
        repository=REPO, revision=REVISION, sources=[str(s) for s in sources], counts=counts,
        camera="image_aux1", format="MP4 bytes plus 30x4 action arrays",
        note="Source-order subset of community-reencoded BAIR; not the full raw-pixel release."
    ), indent=2))
    print("Saved:", final, flush=True)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/parquet/bair_hf_subset"))
    parser.add_argument("--train-clips", type=int, default=1000)
    parser.add_argument("--test-clips", type=int, default=256)
    parser.add_argument("--shards", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--local-tars", nargs="+", help="Optional existing tar files instead of URLs")
    args = parser.parse_args()
    convert(args.output, args.train_clips, args.test_clips, args.shards, args.local_tars)

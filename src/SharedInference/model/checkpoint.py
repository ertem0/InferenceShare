"""Explicit, revision-pinned downloads; no PyTorch model is constructed."""

import argparse
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

OLMOE_MODEL_ID = "allenai/OLMoE-1B-7B-0924"
OLMOE_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"


def download_checkpoint(
    model_id: str,
    revision: str,
    destination: str | Path | None = None,
) -> Path:
    """Download config, index and Safetensors weights directly to disk.

    A full commit hash is required for reproducibility. The default destination
    is ~/Models/<model_id>/<revision>. Existing files are reused by the Hub
    downloader. Hub errors propagate; insufficient disk space raises OSError
    before downloading weights. The dry run requires network access.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full lowercase 40-character commit hash")
    parts = model_id.split("/")
    if not 1 <= len(parts) <= 2 or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in parts
    ):
        raise ValueError("model_id must be a repository name or owner/repository")
    target = (
        Path(destination).expanduser()
        if destination is not None
        else Path.home() / "Models" / model_id / revision
    )
    options = {
        "repo_id": model_id,
        "revision": revision,
        "local_dir": target,
        "allow_patterns": ["config.json", "*.safetensors", "*.safetensors.index.json"],
        "max_workers": 1,
    }
    files = snapshot_download(**options, dry_run=True)
    names = {file.filename for file in files}
    if "config.json" not in names or not any(
        name.endswith(".safetensors") for name in names
    ):
        raise ValueError("Checkpoint must contain config.json and Safetensors weights")
    needed = 0
    for file in files:
        if not file.will_download:
            continue
        size = file.file_size
        if size is None:
            raise ValueError(
                f"Cannot check disk space: unknown download size for {file.filename}"
            )
        needed += size
    existing_parent = target
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    # Leave room for metadata and filesystem overhead. Partial files are counted
    # conservatively at full size by the Hub dry run.
    required = needed + (512 * 1024**2 if needed else 0)
    free = shutil.disk_usage(existing_parent).free
    if free < required:
        raise OSError(
            f"Insufficient disk space: need {required / 1024**3:.2f} GiB "
            f"including headroom, have {free / 1024**3:.2f} GiB at {existing_parent}"
        )
    return Path(snapshot_download(**options))


def main() -> None:
    """Download the default OLMoE checkpoint or an explicitly selected model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=OLMOE_MODEL_ID)
    parser.add_argument("--revision", default=OLMOE_REVISION)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    print(download_checkpoint(args.model_id, args.revision, args.destination))


if __name__ == "__main__":
    main()

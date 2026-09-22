from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from SharedInference.model import checkpoint


def test_download_uses_pinned_revision_and_ssd_directory(monkeypatch, tmp_path):
    files = [
        SimpleNamespace(filename=n, file_size=100, will_download=True)
        for n in ("config.json", "model.safetensors")
    ]
    download = Mock(side_effect=[files, str(tmp_path)])
    monkeypatch.setattr(checkpoint, "snapshot_download", download)
    monkeypatch.setattr(
        checkpoint.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**10)
    )
    result = checkpoint.download_checkpoint("owner/model", "a" * 40, tmp_path)
    assert result == tmp_path
    assert download.call_args_list[0].kwargs["dry_run"] is True
    assert download.call_args.kwargs == {
        "repo_id": "owner/model",
        "revision": "a" * 40,
        "local_dir": tmp_path,
        "allow_patterns": ["config.json", "*.safetensors", "*.safetensors.index.json"],
        "max_workers": 1,
    }


def test_insufficient_space_prevents_download(monkeypatch, tmp_path):
    files = [
        SimpleNamespace(filename=n, file_size=10**9, will_download=True)
        for n in ("config.json", "model.safetensors")
    ]
    download = Mock(return_value=files)
    monkeypatch.setattr(checkpoint, "snapshot_download", download)
    monkeypatch.setattr(
        checkpoint.shutil, "disk_usage", lambda p: SimpleNamespace(free=100)
    )
    with pytest.raises(OSError, match="Insufficient disk space"):
        checkpoint.download_checkpoint("owner/model", "a" * 40, tmp_path)
    assert download.call_count == 1


def test_existing_files_do_not_require_duplicate_space(monkeypatch, tmp_path):
    files = [
        SimpleNamespace(filename=n, file_size=10**10, will_download=False)
        for n in ("config.json", "model.safetensors")
    ]
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    target = tmp_path / "Models/owner/model" / ("a" * 40)
    download = Mock(side_effect=[files, str(target)])
    monkeypatch.setattr(checkpoint, "snapshot_download", download)
    monkeypatch.setattr(
        checkpoint.shutil, "disk_usage", lambda p: SimpleNamespace(free=100)
    )
    assert checkpoint.download_checkpoint("owner/model", "a" * 40) == target


@pytest.mark.parametrize("revision", ["main", "abc123", "../bad"])
def test_requires_commit_hash(revision):
    with pytest.raises(ValueError, match="commit hash"):
        checkpoint.download_checkpoint("owner/model", revision)


def test_unknown_download_size_prevents_download(monkeypatch, tmp_path):
    files = [
        SimpleNamespace(filename="config.json", file_size=100, will_download=True),
        SimpleNamespace(
            filename="model.safetensors", file_size=None, will_download=True
        ),
    ]
    download = Mock(return_value=files)
    monkeypatch.setattr(checkpoint, "snapshot_download", download)
    with pytest.raises(ValueError, match="unknown download size for model.safetensors"):
        checkpoint.download_checkpoint("owner/model", "a" * 40, tmp_path)
    assert download.call_count == 1


def test_download_failure_propagates(monkeypatch, tmp_path):
    monkeypatch.setattr(
        checkpoint, "snapshot_download", Mock(side_effect=ConnectionError("offline"))
    )
    with pytest.raises(ConnectionError, match="offline"):
        checkpoint.download_checkpoint("owner/model", "a" * 40, tmp_path)

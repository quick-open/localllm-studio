"""models: scanning, resolve, catalog, and download logic (mocked network)."""

from __future__ import annotations

import os

import pytest

from localllm import models
from localllm.errors import LocalLLMError


def test_list_models_scans_gguf_only(fake_models_dir):
    found = models.list_models(fake_models_dir)
    names = sorted(m.name for m in found)
    assert names == ["alpha-1b.Q4_K_M.gguf", "beta-instruct.gguf"]
    # notes.txt is ignored
    assert all(m.name.endswith(".gguf") for m in found)


def test_list_models_missing_dir_returns_empty(tmp_path):
    assert models.list_models(str(tmp_path / "nope")) == []


def test_modelinfo_stem_and_size(fake_models_dir):
    mi = [m for m in models.list_models(fake_models_dir)
          if m.name == "beta-instruct.gguf"][0]
    assert mi.stem == "beta-instruct"
    assert mi.size == 16


def test_resolve_by_exact_name(fake_models_dir):
    p = models.resolve("beta-instruct.gguf", directory=fake_models_dir)
    assert os.path.basename(p) == "beta-instruct.gguf"
    assert os.path.isabs(p)


def test_resolve_by_name_without_ext(fake_models_dir):
    p = models.resolve("beta-instruct", directory=fake_models_dir)
    assert os.path.basename(p) == "beta-instruct.gguf"


def test_resolve_by_full_path(fake_models_dir):
    full = os.path.join(fake_models_dir, "alpha-1b.Q4_K_M.gguf")
    assert models.resolve(full) == os.path.abspath(full)


def test_resolve_missing_raises_with_guidance(fake_models_dir):
    with pytest.raises(LocalLLMError) as ei:
        models.resolve("ghost", directory=fake_models_dir)
    msg = str(ei.value)
    assert "not found" in msg
    assert "download" in msg  # points the user at the download command


def test_resolve_empty_raises():
    with pytest.raises(LocalLLMError):
        models.resolve("")


def test_catalog_entries_present():
    entries = models.catalog()
    assert entries
    for e in entries:
        assert e.name.endswith(".gguf")
        assert e.url.startswith("https://")
        assert e.approx_size > 0
    dicts = models.catalog_dicts()
    assert dicts[0]["name"] == entries[0].name


def test_find_catalog_entry():
    first = models.catalog()[0]
    assert models.find_catalog_entry(first.name) is first
    assert models.find_catalog_entry(first.name[:-len(".gguf")]) is first
    assert models.find_catalog_entry("nonsense") is None


def test_download_unknown_name_raises(tmp_path):
    with pytest.raises(LocalLLMError):
        models.download("not-in-catalog", dest_dir=str(tmp_path))


def test_download_writes_file_with_mocked_network(tmp_path, monkeypatch):
    """Exercise the download plumbing without real network I/O."""
    entry = models.catalog()[0]
    payload = b"GGUF-FAKE-BYTES" * 100
    calls = []

    class FakeResp:
        def __init__(self, data):
            self._data = data
            self._pos = 0
            self.headers = {"Content-Length": str(len(data))}

        def read(self, n):
            chunk = self._data[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, *a, **kw):
        calls.append(getattr(req, "full_url", req))
        return FakeResp(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    seen = []
    dest = models.download(entry.name, progress=lambda d, t: seen.append((d, t)),
                           dest_dir=str(tmp_path))
    assert os.path.isfile(dest)
    assert open(dest, "rb").read() == payload
    assert calls and calls[0] == entry.url
    assert seen and seen[-1][0] == len(payload)  # progress reached 100%


def test_download_network_error_wrapped(tmp_path, monkeypatch):
    import urllib.error

    def boom(*a, **kw):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    entry = models.catalog()[0]
    with pytest.raises(LocalLLMError):
        models.download(entry.name, dest_dir=str(tmp_path))
    # partial file cleaned up
    assert not any(f.endswith(".part") for f in os.listdir(tmp_path))


def test_human_size():
    assert models.human_size(0) == "0 B"
    assert models.human_size(1536).endswith("KB")
    assert "GB" in models.human_size(2 * 1024 ** 3)

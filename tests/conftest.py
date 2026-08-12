"""Shared fixtures. Every test runs WITHOUT the native llama.cpp library and
WITHOUT any real model file -- an isolated tmp config dir keeps them hermetic.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point guiconfig/models at a throwaway dir so tests never touch real
    user config or models, and never hit the network."""
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("LOCALLLM_CONFIG_DIR", str(cfg))
    # Make sure the Windows branch of config_dir() is not taken on CI.
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    return str(cfg)


@pytest.fixture
def fake_models_dir(tmp_path):
    """A directory containing a couple of fake .gguf files (empty content)."""
    d = tmp_path / "models"
    d.mkdir()
    for name in ("alpha-1b.Q4_K_M.gguf", "beta-instruct.gguf", "notes.txt"):
        (d / name).write_bytes(b"\x00" * 16)
    return str(d)


def make_word_tokenizer():
    """A trivial deterministic tokenizer: one token per whitespace word."""
    return lambda text: len(text.split()) if text else 0

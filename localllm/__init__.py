"""localllm -- run local GGUF LLMs (chat + completions) fully offline.

Public API is deliberately small and import-safe: importing this package (or any
submodule) does NOT import the native ``llama-cpp-python`` library or load any
model.  That heavy work happens lazily inside :class:`~localllm.engine.LlamaEngine`
the first time you actually generate text, so the package, its CLI and its GUI
all import cleanly on a machine with no native library and no model installed.

    from localllm import LlamaEngine, ChatSession, list_models
    eng = LlamaEngine("my-model")          # cheap; no native load yet
    print(eng.complete("Hello"))            # loads llama.cpp here (or raises
                                            # LocalLLMError with guidance)

Every recoverable failure raises :class:`LocalLLMError` and nothing else.
"""

from __future__ import annotations

from .errors import LocalLLMError
from .engine import LlamaEngine, DEFAULT_PARAMS, library_available
from .chatsession import ChatSession, PRESETS, preset_names, get_preset
from .models import (
    ModelInfo,
    CatalogEntry,
    list_models,
    resolve,
    catalog,
    catalog_dicts,
    find_catalog_entry,
    download,
    models_dir,
    human_size,
)
from . import prompts

__version__ = "1.0.5"

__all__ = [
    "LocalLLMError",
    "LlamaEngine",
    "DEFAULT_PARAMS",
    "library_available",
    "ChatSession",
    "PRESETS",
    "preset_names",
    "get_preset",
    "ModelInfo",
    "CatalogEntry",
    "list_models",
    "resolve",
    "catalog",
    "catalog_dicts",
    "find_catalog_entry",
    "download",
    "models_dir",
    "human_size",
    "prompts",
    "__version__",
]

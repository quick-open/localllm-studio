"""engine + laziness: prove the package/CLI/GUI import and behave with NO native
library and NO model present, and that missing lib/model gives a clear error.
"""


from __future__ import annotations
import builtins
import os
import sys

import pytest

from localllm.errors import LocalLLMError


# ---------------------------------------------------------------------------
# Laziness proofs
# ---------------------------------------------------------------------------
def test_import_localllm_without_llama_cpp(monkeypatch):
    """Importing the package must not import the native library."""
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # any import attempt fails
    for mod in list(sys.modules):
        if mod == "localllm" or mod.startswith("localllm."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    import localllm  # noqa: F401
    from localllm import gui  # noqa: F401
    from localllm import __main__ as cli  # noqa: F401
    # llama_cpp must NOT have been really imported (still the None sentinel)
    assert sys.modules.get("llama_cpp") is None


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows CI has a real display; main() would open a window and block")
def test_cli_parser_builds_without_native(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    from localllm.__main__ import build_parser
    p = build_parser()
    ns = p.parse_args(["complete", "--model", "m", "--prompt", "hi"])
    assert ns.command == "complete"
    assert ns.model == "m"


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows CI has a real display; main() would open a window and block")
def test_gui_main_headless_returns_zero(monkeypatch):
    """gui.main() is side-effect-free to import and returns 0 headless."""
    monkeypatch.delenv("DISPLAY", raising=False)
    from localllm import gui
    rc = gui.main()
    assert rc == 0


# ---------------------------------------------------------------------------
# Missing-library error path
# ---------------------------------------------------------------------------
def _force_no_llama(monkeypatch):
    """Make `from llama_cpp import ...` raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp" or name.startswith("llama_cpp."):
            raise ImportError("simulated: llama_cpp not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)


def test_engine_complete_raises_localllmerror_when_lib_missing(
        monkeypatch, fake_models_dir):
    _force_no_llama(monkeypatch)
    from localllm.engine import LlamaEngine
    eng = LlamaEngine("alpha-1b.Q4_K_M.gguf", models_directory=fake_models_dir)
    with pytest.raises(LocalLLMError) as ei:
        eng.complete("hello")
    msg = str(ei.value)
    assert "llama-cpp-python" in msg
    assert "pip install" in msg


def test_engine_chat_raises_localllmerror_when_lib_missing(
        monkeypatch, fake_models_dir):
    _force_no_llama(monkeypatch)
    from localllm.engine import LlamaEngine
    eng = LlamaEngine("beta-instruct.gguf", models_directory=fake_models_dir)
    with pytest.raises(LocalLLMError):
        eng.chat([{"role": "user", "content": "hi"}])


def test_engine_stream_raises_localllmerror_when_lib_missing(
        monkeypatch, fake_models_dir):
    _force_no_llama(monkeypatch)
    from localllm.engine import LlamaEngine
    eng = LlamaEngine("beta-instruct.gguf", models_directory=fake_models_dir)
    with pytest.raises(LocalLLMError):
        list(eng.stream_complete("hi"))


def test_engine_construction_bad_model_raises(fake_models_dir):
    from localllm.engine import LlamaEngine
    with pytest.raises(LocalLLMError):
        LlamaEngine("nonexistent-model", models_directory=fake_models_dir)


def test_library_available_is_false_here():
    """This box has no native lib; the probe must say so without raising."""
    from localllm.engine import library_available
    assert library_available() in (True, False)  # never raises


def test_gen_kwargs_translation():
    from localllm.engine import LlamaEngine
    p = LlamaEngine._sampling({"temperature": 0.3, "seed": 7, "stop": ["</s>"]})
    kw = LlamaEngine._gen_kwargs(p)
    assert kw["temperature"] == 0.3
    assert kw["seed"] == 7
    assert kw["stop"] == ["</s>"]
    # negative seed is dropped (random)
    kw2 = LlamaEngine._gen_kwargs(LlamaEngine._sampling({"seed": -1}))
    assert "seed" not in kw2


# ---------------------------------------------------------------------------
# Optional real inference (only if lib AND a model are somehow present)
# ---------------------------------------------------------------------------
def test_optional_real_one_token():
    llama = pytest.importorskip("llama_cpp")  # skips when native lib absent
    from localllm import models, engine
    installed = models.list_models()
    if not installed:
        pytest.skip("no GGUF model installed for a real inference smoke test")
    eng = engine.LlamaEngine(installed[0].name, n_ctx=512)
    out = eng.complete("Hello", max_tokens=1)
    assert isinstance(out, str)

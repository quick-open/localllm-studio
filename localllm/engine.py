r"""The LLM inference engine -- a thin, well-guarded wrapper over llama.cpp.

The single most important property of this module: **the native library
(``llama_cpp``) is imported lazily**, only inside :meth:`LlamaEngine._ensure`,
never at module import time.  That means ``import localllm.engine`` (and the
whole package, the CLI parser and the GUI) works on a machine that has neither
``llama-cpp-python`` nor any GGUF model installed -- exactly the environment the
test-suite runs in.  A missing library or model surfaces as a clear
:class:`LocalLLMError` with guidance, only when you actually try to generate.
"""

from __future__ import annotations

from . import models as _models
from .errors import LocalLLMError

# Default sampling parameters; the GUI/CLI override individual keys.
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_tokens": 512,
    "repeat_penalty": 1.1,
    "seed": -1,
    "stop": None,
}

_LIB_MISSING_MSG = (
    "The native llama.cpp library (llama-cpp-python) is not installed.\n"
    "  Install it with:  pip install llama-cpp-python\n"
    "  (a prebuilt wheel or a working C/C++ toolchain is required).\n"
    "  Everything else in LocalLLM Studio works without it -- you only need it "
    "to actually run a model."
)


def _import_llama():
    """Import and return the ``Llama`` class, or raise a friendly error.

    Kept in its own function so tests can monkeypatch ``llama_cpp`` in
    ``sys.modules`` to simulate the library being absent.
    """
    try:
        from llama_cpp import Llama  # noqa: WPS433 - intentionally lazy/local
    except Exception as exc:  # ImportError or a broken native build
        raise LocalLLMError(f"{_LIB_MISSING_MSG}\n  (import error: {exc})")
    return Llama


class LlamaEngine:
    """Lazily-loaded llama.cpp model wrapper.

    Construction is cheap and does no native work: it only resolves and
    remembers the model path and load options.  The heavy ``Llama(...)`` load
    happens on first use (or an explicit :meth:`load`).
    """

    def __init__(self, model, n_ctx=4096, n_threads=None, n_gpu_layers=0,
                 chat_format=None, verbose=False, models_directory=None):
        # Resolve eagerly so a bad name fails fast with a helpful message,
        # but do NOT touch llama_cpp here.
        self.model_path = _models.resolve(model, directory=models_directory)
        self.n_ctx = int(n_ctx)
        self.n_threads = n_threads
        self.n_gpu_layers = int(n_gpu_layers)
        self.chat_format = chat_format
        self.verbose = bool(verbose)
        self._llm = None

    # -- loading ---------------------------------------------------------
    def is_loaded(self):
        return self._llm is not None

    def load(self):
        """Force the native model load now. Idempotent."""
        self._ensure()
        return self

    def _ensure(self):
        if self._llm is not None:
            return self._llm
        Llama = _import_llama()
        kwargs = dict(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )
        if self.n_threads:
            kwargs["n_threads"] = int(self.n_threads)
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format
        try:
            self._llm = Llama(**kwargs)
        except LocalLLMError:
            raise
        except Exception as exc:
            raise LocalLLMError(
                f"Failed to load model {self.model_path!r}: {exc}\n"
                f"  The file may be missing, corrupt, or an unsupported GGUF "
                f"version. Try re-downloading it."
            )
        return self._llm

    def close(self):
        """Release the native model (best effort)."""
        self._llm = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _sampling(params):
        p = dict(DEFAULT_PARAMS)
        p.update({k: v for k, v in (params or {}).items() if v is not None})
        return p

    @staticmethod
    def _gen_kwargs(p):
        """Translate our param names to llama.cpp create_* kwargs."""
        kwargs = dict(
            temperature=float(p["temperature"]),
            top_p=float(p["top_p"]),
            top_k=int(p["top_k"]),
            max_tokens=int(p["max_tokens"]),
            repeat_penalty=float(p["repeat_penalty"]),
        )
        seed = p.get("seed")
        if seed is not None and int(seed) >= 0:
            kwargs["seed"] = int(seed)
        stop = p.get("stop")
        if stop:
            kwargs["stop"] = list(stop) if not isinstance(stop, str) else [stop]
        return kwargs

    # -- text completion -------------------------------------------------
    def complete(self, prompt, **params):
        """Return the completion text for *prompt* (blocking)."""
        llm = self._ensure()
        kwargs = self._gen_kwargs(self._sampling(params))
        try:
            out = llm(prompt, stream=False, **kwargs)
            return out["choices"][0]["text"]
        except LocalLLMError:
            raise
        except Exception as exc:
            raise LocalLLMError(f"Completion failed: {exc}")

    def stream_complete(self, prompt, **params):
        """Yield completion text chunks for *prompt* as they are produced."""
        llm = self._ensure()
        kwargs = self._gen_kwargs(self._sampling(params))
        try:
            for chunk in llm(prompt, stream=True, **kwargs):
                piece = chunk["choices"][0].get("text", "")
                if piece:
                    yield piece
        except LocalLLMError:
            raise
        except Exception as exc:
            raise LocalLLMError(f"Streaming completion failed: {exc}")

    # -- chat ------------------------------------------------------------
    def chat(self, messages, **params):
        """Return the assistant reply text for a list of chat *messages*.

        *messages* is a list of ``{"role": ..., "content": ...}`` dicts.
        """
        llm = self._ensure()
        kwargs = self._gen_kwargs(self._sampling(params))
        try:
            out = llm.create_chat_completion(
                messages=list(messages), stream=False, **kwargs)
            return out["choices"][0]["message"]["content"]
        except LocalLLMError:
            raise
        except Exception as exc:
            raise LocalLLMError(f"Chat failed: {exc}")

    def stream_chat(self, messages, **params):
        """Yield assistant reply chunks for a list of chat *messages*."""
        llm = self._ensure()
        kwargs = self._gen_kwargs(self._sampling(params))
        try:
            for chunk in llm.create_chat_completion(
                    messages=list(messages), stream=True, **kwargs):
                delta = chunk["choices"][0].get("delta", {})
                piece = delta.get("content", "")
                if piece:
                    yield piece
        except LocalLLMError:
            raise
        except Exception as exc:
            raise LocalLLMError(f"Streaming chat failed: {exc}")


def library_available():
    """True if the native llama.cpp library can be imported right now.

    Never raises; the GUI uses this to show a friendly banner.
    """
    try:
        _import_llama()
        return True
    except Exception:
        return False

"""Error types for localllm."""


class LocalLLMError(Exception):
    """Raised for any recoverable failure in a localllm operation.

    All public functions raise this (and only this) on failure so callers
    -- including the CLI and the GUI -- have a single exception to catch.
    Typical causes: the native ``llama-cpp-python`` library is not installed,
    no GGUF model is available, or a model file cannot be loaded.
    """

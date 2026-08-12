r"""Model management for LocalLLM Studio.

GGUF model files live in the app's models directory
(``%LOCALAPPDATA%\LocalLLMStudio\models`` on Windows, ``~/.localllmstudio/models``
otherwise).  This module lists installed models, resolves a name-or-path to a
concrete file, exposes a small curated catalog of known-good *permissively*
licensed GGUF models, and can download a catalog entry on explicit user request.

Network access happens ONLY inside :func:`download` -- listing, resolving and
reading the catalog are entirely offline.
"""

from __future__ import annotations

import os

from . import guiconfig
from .errors import LocalLLMError

GGUF_EXT = ".gguf"


class ModelInfo:
    """A GGUF model file that exists on disk."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path)
        try:
            self.size = os.path.getsize(self.path)
        except OSError:
            self.size = 0

    @property
    def stem(self):
        """File name without the ``.gguf`` extension."""
        base = self.name
        if base.lower().endswith(GGUF_EXT):
            base = base[: -len(GGUF_EXT)]
        return base

    def __repr__(self):
        return f"ModelInfo(name={self.name!r}, size={self.size})"


# ---------------------------------------------------------------------------
# Curated catalog -- small, permissively licensed GGUF models.
# Sizes are approximate (bytes) and only used for display/progress hints.
# ---------------------------------------------------------------------------
class CatalogEntry:
    def __init__(self, name, params, quant, approx_size, license_, url, note=""):
        self.name = name          # the filename it installs as (…gguf)
        self.params = params      # e.g. "1.1B"
        self.quant = quant        # e.g. "Q4_K_M"
        self.approx_size = approx_size  # bytes
        self.license = license_
        self.url = url
        self.note = note

    def as_dict(self):
        return {
            "name": self.name,
            "params": self.params,
            "quant": self.quant,
            "approx_size": self.approx_size,
            "license": self.license,
            "url": self.url,
            "note": self.note,
        }


CATALOG = [
    CatalogEntry(
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        params="1.1B", quant="Q4_K_M", approx_size=669_000_000,
        license_="Apache-2.0",
        url=(
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/"
            "resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        ),
        note="Tiny, fast, great for smoke-testing on CPU.",
    ),
    CatalogEntry(
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        params="0.5B", quant="Q4_K_M", approx_size=398_000_000,
        license_="Apache-2.0",
        url=(
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/"
            "resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ),
        note="Very small Qwen2.5 instruct model.",
    ),
    CatalogEntry(
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        params="1.5B", quant="Q4_K_M", approx_size=986_000_000,
        license_="Apache-2.0",
        url=(
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/"
            "resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        ),
        note="A capable small chat model, still CPU-friendly.",
    ),
    CatalogEntry(
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        params="1B", quant="Q4_K_M", approx_size=808_000_000,
        license_="Llama-3.2-Community",
        url=(
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/"
            "resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        ),
        note="Meta Llama 3.2 1B (community license -- review before use).",
    ),
]


def catalog():
    """Return the curated catalog as a list of :class:`CatalogEntry`."""
    return list(CATALOG)


def catalog_dicts():
    """Catalog as plain dicts (handy for the CLI/GUI and tests)."""
    return [e.as_dict() for e in CATALOG]


def find_catalog_entry(name):
    """Look up a catalog entry by exact name or by name without ``.gguf``.

    Returns the entry or ``None``.
    """
    if not name:
        return None
    lname = name.lower()
    for e in CATALOG:
        if e.name.lower() == lname or e.name.lower() == lname + GGUF_EXT:
            return e
    return None


# ---------------------------------------------------------------------------
# Installed models
# ---------------------------------------------------------------------------
def models_dir(create=False):
    """The directory that holds installed GGUF files."""
    d = guiconfig.models_dir()
    if create:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as exc:
            raise LocalLLMError(f"Could not create models directory {d}: {exc}")
    return d


def list_models(directory=None):
    """List installed GGUF models (``*.gguf``) in *directory*.

    Defaults to the app models directory.  A missing directory yields an empty
    list rather than an error.  Results are sorted by name.
    """
    d = directory or models_dir()
    out = []
    try:
        entries = os.listdir(d)
    except OSError:
        return out
    for name in sorted(entries):
        if name.lower().endswith(GGUF_EXT):
            full = os.path.join(d, name)
            if os.path.isfile(full):
                out.append(ModelInfo(full))
    return out


def resolve(name_or_path, directory=None):
    """Resolve *name_or_path* to an absolute GGUF file path.

    Accepts, in order of preference:
      * an existing file path (absolute or relative to CWD);
      * a bare model name (with or without ``.gguf``) in the models directory.

    Raises :class:`LocalLLMError` with guidance if nothing matches.
    """
    if not name_or_path:
        raise LocalLLMError("No model specified.")

    # 1) a real path the user handed us
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)

    d = directory or models_dir()
    candidates = [name_or_path]
    if not name_or_path.lower().endswith(GGUF_EXT):
        candidates.append(name_or_path + GGUF_EXT)
    for cand in candidates:
        full = os.path.join(d, cand)
        if os.path.isfile(full):
            return os.path.abspath(full)

    # 2) case-insensitive match against installed models
    lname = name_or_path.lower()
    for mi in list_models(d):
        if mi.name.lower() == lname or mi.stem.lower() == lname:
            return mi.path

    installed = ", ".join(m.name for m in list_models(d)) or "(none installed)"
    raise LocalLLMError(
        f"Model {name_or_path!r} not found.\n"
        f"  Looked for a file at that path and in {d}\n"
        f"  Installed models: {installed}\n"
        f"  Put a .gguf file in the models directory, pass a full path, or run "
        f"`localllm models download <name>`."
    )


# ---------------------------------------------------------------------------
# Explicit, user-triggered download
# ---------------------------------------------------------------------------
def download(name, progress=None, dest_dir=None, url=None, chunk=1 << 20):
    """Download a catalog model into the models directory.

    *name* must match a catalog entry (unless *url* is given explicitly).
    *progress*, if provided, is called as ``progress(done_bytes, total_bytes)``
    where ``total_bytes`` may be 0 if the server does not send a length.

    Returns the absolute path to the downloaded file.  This is the ONLY function
    in the package that touches the network, and only when the user asks for it.
    """
    entry = find_catalog_entry(name)
    if url is None:
        if entry is None:
            raise LocalLLMError(
                f"{name!r} is not in the catalog. Known models: "
                f"{', '.join(e.name for e in CATALOG)}"
            )
        url = entry.url
    filename = entry.name if entry else os.path.basename(name)
    if not filename.lower().endswith(GGUF_EXT):
        filename += GGUF_EXT

    d = dest_dir or models_dir(create=True)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as exc:
        raise LocalLLMError(f"Could not create {d}: {exc}")
    dest = os.path.join(d, filename)
    tmp = dest + ".part"

    # urllib is stdlib -- no hard dependency on requests.
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LocalLLMStudio"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310 - explicit user action
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if progress:
                        try:
                            progress(done, total)
                        except Exception:
                            pass
        os.replace(tmp, dest)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise LocalLLMError(f"Download of {filename} failed: {exc}")
    return os.path.abspath(dest)


def human_size(num_bytes):
    """Human-readable byte size."""
    size = float(num_bytes or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

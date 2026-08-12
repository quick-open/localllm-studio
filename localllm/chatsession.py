"""Conversation state for a chat: system prompt, message history, presets,
token-budget trimming, and JSON/Markdown export/import.

Pure Python and dependency-free -- it never imports the native LLM library, so
it is fully unit-testable on any box.  Token accounting uses an injectable
*tokenizer* (a callable ``text -> int``); the default is a cheap word/char
heuristic so trimming works without a real model.
"""

from __future__ import annotations

import json
import time

from .errors import LocalLLMError

ROLES = ("system", "user", "assistant")


# ---------------------------------------------------------------------------
# Parameter presets -- a name, a system-prompt preset name, and sampling params.
# (System-prompt *text* lives in prompts.py; here we bundle sampling defaults.)
# ---------------------------------------------------------------------------
PRESETS = {
    "assistant": {
        "prompt": "Assistant",
        "params": {"temperature": 0.7, "top_p": 0.95, "top_k": 40,
                   "repeat_penalty": 1.1, "max_tokens": 512},
    },
    "coding": {
        "prompt": "Coding",
        "params": {"temperature": 0.2, "top_p": 0.9, "top_k": 40,
                   "repeat_penalty": 1.05, "max_tokens": 1024},
    },
    "creative": {
        "prompt": "Creative",
        "params": {"temperature": 1.0, "top_p": 0.98, "top_k": 100,
                   "repeat_penalty": 1.15, "max_tokens": 768},
    },
}


def preset_names():
    return list(PRESETS.keys())


def get_preset(name):
    """Return a deep-ish copy of a preset dict, or raise LocalLLMError."""
    p = PRESETS.get(name)
    if p is None:
        raise LocalLLMError(
            f"Unknown preset {name!r}. Available: {', '.join(PRESETS)}")
    return {"prompt": p["prompt"], "params": dict(p["params"])}


def default_tokenizer(text):
    """Cheap token estimate: ~1 token per 4 chars, but at least the word count.

    Good enough for history trimming without loading a real tokenizer.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(words, (chars + 3) // 4)


class ChatSession:
    """Holds a system prompt plus an ordered list of user/assistant turns."""

    def __init__(self, system=None, params=None, tokenizer=None,
                 max_context_tokens=4096, model=None):
        self.system = system or ""
        self.params = dict(params or {})
        self.tokenizer = tokenizer or default_tokenizer
        self.max_context_tokens = int(max_context_tokens)
        self.model = model or ""
        self.messages = []  # list of {"role","content"} (no system entries)
        self.created = time.time()

    # -- system prompt ---------------------------------------------------
    def set_system(self, text):
        self.system = text or ""

    # -- history ---------------------------------------------------------
    def add_user(self, content):
        return self._add("user", content)

    def add_assistant(self, content):
        return self._add("assistant", content)

    def _add(self, role, content):
        if role not in ("user", "assistant"):
            raise LocalLLMError(f"Cannot add message with role {role!r}.")
        msg = {"role": role, "content": content or ""}
        self.messages.append(msg)
        return msg

    def reset(self):
        """Clear the message history (keep system prompt & params)."""
        self.messages = []

    def clear(self):
        self.reset()

    # -- building the model-facing message list --------------------------
    def build_messages(self, trim=True):
        """Return the full message list (system + trimmed history) to send.

        When *trim* is True, the oldest user/assistant turns are dropped (the
        system prompt is always kept) until the estimated token count fits in
        ``max_context_tokens`` minus a reservation for ``params['max_tokens']``.
        """
        history = self.messages
        if trim:
            history = self._trim(history)
        out = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.extend({"role": m["role"], "content": m["content"]} for m in history)
        return out

    def token_count(self, messages=None):
        """Estimated tokens for *messages* (defaults to system + full history)."""
        if messages is None:
            messages = ([{"role": "system", "content": self.system}]
                        if self.system else []) + self.messages
        total = 0
        for m in messages:
            # small per-message overhead for role/formatting tokens
            total += self.tokenizer(m.get("content", "")) + 4
        return total

    def _trim(self, history):
        reserve = int(self.params.get("max_tokens", 0) or 0)
        budget = max(0, self.max_context_tokens - reserve)
        sys_tokens = (self.tokenizer(self.system) + 4) if self.system else 0

        # Keep dropping the oldest turn until we fit (always keep >= last turn).
        kept = list(history)
        while len(kept) > 1:
            total = sys_tokens + sum(self.tokenizer(m["content"]) + 4 for m in kept)
            if total <= budget:
                break
            kept.pop(0)
        return kept

    # -- presets ---------------------------------------------------------
    def apply_preset(self, name, prompts_module=None):
        """Apply a named preset: set params and (if resolvable) system prompt."""
        preset = get_preset(name)
        self.params.update(preset["params"])
        # Resolve the system-prompt text lazily to avoid an import cycle at top.
        if prompts_module is None:
            from . import prompts as prompts_module
        try:
            self.system = prompts_module.get_prompt(preset["prompt"])
        except LocalLLMError:
            pass  # a missing custom preset shouldn't break param application
        return self

    # -- serialization ---------------------------------------------------
    def to_dict(self):
        return {
            "app": "LocalLLM Studio",
            "version": 1,
            "model": self.model,
            "system": self.system,
            "params": dict(self.params),
            "messages": [dict(m) for m in self.messages],
            "created": self.created,
        }

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self):
        lines = ["# LocalLLM Studio chat", ""]
        if self.model:
            lines.append(f"*Model:* `{self.model}`")
        if self.system:
            lines += ["", "**System prompt**", "", "> " +
                      self.system.replace("\n", "\n> "), ""]
        for m in self.messages:
            who = "You" if m["role"] == "user" else "Assistant"
            lines.append(f"### {who}")
            lines.append("")
            lines.append(m["content"])
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def save(self, path, fmt=None):
        """Write the chat to *path* as JSON (.json) or Markdown (.md)."""
        fmt = (fmt or self._fmt_from_path(path)).lower()
        try:
            with open(path, "w", encoding="utf-8") as fh:
                if fmt == "md":
                    fh.write(self.to_markdown())
                else:
                    fh.write(self.to_json())
        except OSError as exc:
            raise LocalLLMError(f"Could not write chat to {path}: {exc}")
        return path

    @staticmethod
    def _fmt_from_path(path):
        return "md" if str(path).lower().endswith((".md", ".markdown")) else "json"

    # -- import ----------------------------------------------------------
    @classmethod
    def from_dict(cls, data, tokenizer=None, max_context_tokens=4096):
        if not isinstance(data, dict):
            raise LocalLLMError("Chat data must be a JSON object.")
        sess = cls(
            system=data.get("system", ""),
            params=data.get("params") or {},
            tokenizer=tokenizer,
            max_context_tokens=max_context_tokens,
            model=data.get("model", ""),
        )
        msgs = data.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role")
                content = m.get("content", "")
                if role in ("user", "assistant"):
                    sess.messages.append({"role": role, "content": content})
        created = data.get("created")
        if isinstance(created, (int, float)):
            sess.created = created
        return sess

    @classmethod
    def from_json(cls, text, **kw):
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise LocalLLMError(f"Invalid chat JSON: {exc}")
        return cls.from_dict(data, **kw)

    @classmethod
    def load(cls, path, **kw):
        """Load a chat previously saved as JSON (Markdown is export-only)."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise LocalLLMError(f"Could not read chat from {path}: {exc}")
        return cls.from_json(text, **kw)

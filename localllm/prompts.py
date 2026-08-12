"""A small library of reusable system-prompt presets.

Built-in presets ship with the app; users can save their own to the GUI config
(``guiconfig.custom_prompts``).  Nothing here touches the network or the native
LLM library -- it is pure data plus thin config helpers, so it imports and tests
freely on any box.
"""

from __future__ import annotations

from . import guiconfig
from .errors import LocalLLMError

# name -> system prompt text
BUILTIN_PROMPTS = {
    "Assistant": (
        "You are a helpful, honest and concise assistant. Answer clearly and "
        "admit when you are unsure."
    ),
    "Coding": (
        "You are an expert software engineer. Give correct, idiomatic code with "
        "brief explanations. Prefer standard-library solutions and point out "
        "edge cases."
    ),
    "Creative": (
        "You are a vivid, imaginative writing partner. Write with rich detail "
        "and a strong voice while following the user's intent."
    ),
    "Concise": (
        "You are a terse assistant. Answer in as few words as possible while "
        "remaining correct. No preamble."
    ),
    "Socratic": (
        "You are a patient tutor. Guide the user to the answer with clarifying "
        "questions and small steps rather than simply stating the solution."
    ),
}

DEFAULT_PROMPT_NAME = "Assistant"


def builtin_names():
    """Names of the built-in presets, in a stable order."""
    return list(BUILTIN_PROMPTS.keys())


def custom_names():
    """Names of the user's saved custom presets."""
    return sorted(guiconfig.get_custom_prompts().keys())


def all_names():
    """Built-in names followed by custom names (no duplicates dropped -- a
    custom preset may intentionally shadow a built-in by using the same name)."""
    names = builtin_names()
    for n in custom_names():
        if n not in names:
            names.append(n)
    return names


def get_prompt(name):
    """Return the system-prompt text for *name*.

    Custom presets take precedence over built-ins of the same name.  Raises
    :class:`LocalLLMError` if the name is unknown.
    """
    custom = guiconfig.get_custom_prompts()
    if name in custom:
        return custom[name]
    if name in BUILTIN_PROMPTS:
        return BUILTIN_PROMPTS[name]
    raise LocalLLMError(
        f"Unknown prompt preset: {name!r}. "
        f"Available: {', '.join(all_names())}"
    )


def default_prompt():
    """The default system prompt text."""
    return BUILTIN_PROMPTS[DEFAULT_PROMPT_NAME]


def save_custom(name, text):
    """Save (or overwrite) a custom preset. Raises on an empty name."""
    name = (name or "").strip()
    if not name:
        raise LocalLLMError("A preset needs a non-empty name.")
    guiconfig.set_custom_prompt(name, text or "")


def delete_custom(name):
    """Delete a custom preset. Built-ins cannot be deleted."""
    if name not in guiconfig.get_custom_prompts():
        raise LocalLLMError(f"No custom preset named {name!r}.")
    guiconfig.delete_custom_prompt(name)


def is_builtin(name):
    return name in BUILTIN_PROMPTS

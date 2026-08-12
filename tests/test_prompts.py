"""prompts: preset lookup + custom save/load/delete (to a tmp config)."""

from __future__ import annotations

import pytest

from localllm import prompts
from localllm.errors import LocalLLMError


def test_builtin_lookup():
    assert "Assistant" in prompts.builtin_names()
    text = prompts.get_prompt("Assistant")
    assert isinstance(text, str) and text


def test_default_prompt():
    assert prompts.default_prompt() == prompts.get_prompt("Assistant")


def test_unknown_prompt_raises():
    with pytest.raises(LocalLLMError):
        prompts.get_prompt("no-such-preset")


def test_custom_save_and_load():
    prompts.save_custom("Pirate", "Talk like a pirate.")
    assert "Pirate" in prompts.custom_names()
    assert prompts.get_prompt("Pirate") == "Talk like a pirate."
    assert "Pirate" in prompts.all_names()


def test_custom_overrides_builtin_name():
    prompts.save_custom("Assistant", "custom override")
    assert prompts.get_prompt("Assistant") == "custom override"


def test_custom_delete():
    prompts.save_custom("Temp", "temporary")
    prompts.delete_custom("Temp")
    assert "Temp" not in prompts.custom_names()
    with pytest.raises(LocalLLMError):
        prompts.delete_custom("Temp")  # already gone


def test_empty_name_rejected():
    with pytest.raises(LocalLLMError):
        prompts.save_custom("   ", "text")


def test_is_builtin():
    assert prompts.is_builtin("Coding")
    assert not prompts.is_builtin("Pirate")

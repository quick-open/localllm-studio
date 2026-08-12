"""ChatSession: history, system prompt, trimming, export/import, presets."""

from __future__ import annotations

import json

import pytest

from localllm.chatsession import ChatSession, get_preset, preset_names
from localllm.errors import LocalLLMError
from tests.conftest import make_word_tokenizer


def test_history_append_and_roles():
    s = ChatSession(system="sys")
    s.add_user("hi")
    s.add_assistant("hello")
    assert [m["role"] for m in s.messages] == ["user", "assistant"]
    assert s.messages[0]["content"] == "hi"


def test_bad_role_raises():
    s = ChatSession()
    with pytest.raises(LocalLLMError):
        s._add("system", "nope")


def test_system_prompt_in_build_messages():
    s = ChatSession(system="be nice")
    s.add_user("q")
    built = s.build_messages()
    assert built[0] == {"role": "system", "content": "be nice"}
    assert built[-1] == {"role": "user", "content": "q"}


def test_no_system_prompt_omitted():
    s = ChatSession(system="")
    s.add_user("q")
    built = s.build_messages()
    assert all(m["role"] != "system" for m in built)


def test_token_budget_trimming_drops_oldest():
    tok = make_word_tokenizer()
    # tiny budget so only the most recent turns survive
    s = ChatSession(system="sys word here", tokenizer=tok,
                    max_context_tokens=20, params={"max_tokens": 5})
    for i in range(10):
        s.add_user(f"message number {i} padding padding padding")
    built = s.build_messages(trim=True)
    # system always kept; history trimmed below the full 10
    assert built[0]["role"] == "system"
    history = [m for m in built if m["role"] != "system"]
    assert 0 < len(history) < 10
    # the LAST message is always retained
    assert history[-1]["content"].endswith("padding padding padding")
    assert "9" in history[-1]["content"]


def test_trimming_keeps_at_least_one_even_if_over_budget():
    tok = make_word_tokenizer()
    s = ChatSession(system="", tokenizer=tok, max_context_tokens=1)
    s.add_user("a very long message that exceeds the tiny budget by far")
    built = s.build_messages(trim=True)
    assert len(built) == 1


def test_export_import_json_roundtrip():
    s = ChatSession(system="sysprompt", params={"temperature": 0.5}, model="m.gguf")
    s.add_user("one")
    s.add_assistant("two")
    text = s.to_json()
    data = json.loads(text)
    assert data["messages"][0]["content"] == "one"
    back = ChatSession.from_json(text)
    assert back.system == "sysprompt"
    assert back.model == "m.gguf"
    assert [m["content"] for m in back.messages] == ["one", "two"]
    assert back.params["temperature"] == 0.5


def test_save_load_json_file(tmp_path):
    s = ChatSession(system="sp")
    s.add_user("hi")
    s.add_assistant("yo")
    p = tmp_path / "chat.json"
    s.save(str(p))
    back = ChatSession.load(str(p))
    assert [m["content"] for m in back.messages] == ["hi", "yo"]


def test_markdown_export_contains_turns(tmp_path):
    s = ChatSession(system="sp", model="x.gguf")
    s.add_user("question?")
    s.add_assistant("answer.")
    md = s.to_markdown()
    assert "question?" in md and "answer." in md
    assert "### You" in md and "### Assistant" in md
    p = tmp_path / "chat.md"
    s.save(str(p))
    assert p.read_text(encoding="utf-8").strip().endswith("answer.")


def test_from_json_invalid_raises():
    with pytest.raises(LocalLLMError):
        ChatSession.from_json("{not json")


def test_presets_apply_params():
    for name in preset_names():
        preset = get_preset(name)
        assert "params" in preset and "temperature" in preset["params"]
    s = ChatSession()
    s.apply_preset("coding")
    assert s.params["temperature"] == 0.2
    # system prompt got resolved from the prompts module
    assert "engineer" in s.system.lower()


def test_apply_unknown_preset_raises():
    s = ChatSession()
    with pytest.raises(LocalLLMError):
        s.apply_preset("does-not-exist")


def test_reset_clears_history_keeps_system():
    s = ChatSession(system="keepme")
    s.add_user("a")
    s.reset()
    assert s.messages == []
    assert s.system == "keepme"

r"""Tiny JSON-backed config for LocalLLM Studio.

Stores the GUI theme, the last-used model, sampling parameters, and any custom
system-prompt presets the user saves.  On Windows the file lives at
``%LOCALAPPDATA%\LocalLLMStudio\config.json``; elsewhere it falls back to
``~/.localllmstudio/config.json``.  Every function is defensive -- a corrupt or
unreadable config must never stop the app from starting.
"""

from __future__ import annotations

import json
import os

APP_DIRNAME = "LocalLLMStudio"
CONFIG_NAME = "config.json"
VALID_THEMES = ("light", "dark")

# Default sampling parameters (mirrored by engine.DEFAULT_PARAMS).
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_tokens": 512,
    "repeat_penalty": 1.1,
    "seed": -1,
}


def config_dir():
    r"""Directory that holds the config file (created on demand).

    ``%LOCALAPPDATA%\LocalLLMStudio`` on Windows, ``~/.localllmstudio`` otherwise.
    """
    local = os.environ.get("LOCALAPPDATA")
    if local and os.name == "nt":
        return os.path.join(local, APP_DIRNAME)
    # Honour an override so tests can point at a tmp dir without touching HOME.
    override = os.environ.get("LOCALLLM_CONFIG_DIR")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), "." + APP_DIRNAME.lower())


def config_path():
    return os.path.join(config_dir(), CONFIG_NAME)


def models_dir():
    r"""Directory where GGUF models live: ``<config_dir>\models``."""
    d = os.path.join(config_dir(), "models")
    return d


def _defaults():
    return {
        "theme": "light",
        "last_model": "",
        "params": dict(DEFAULT_PARAMS),
        "custom_prompts": {},
    }


def load():
    """Return the config dict, always with every documented key present."""
    cfg = _defaults()
    try:
        with open(config_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            theme = data.get("theme")
            if theme in VALID_THEMES:
                cfg["theme"] = theme
            last = data.get("last_model")
            if isinstance(last, str):
                cfg["last_model"] = last
            params = data.get("params")
            if isinstance(params, dict):
                merged = dict(DEFAULT_PARAMS)
                for k in DEFAULT_PARAMS:
                    if k in params and isinstance(params[k], (int, float)):
                        merged[k] = params[k]
                cfg["params"] = merged
            custom = data.get("custom_prompts")
            if isinstance(custom, dict):
                cfg["custom_prompts"] = {
                    str(k): str(v) for k, v in custom.items()
                    if isinstance(v, str)
                }
    except Exception:
        pass  # missing/corrupt -> defaults; never fatal
    return cfg


def save(cfg):
    """Persist *cfg* (best-effort; failures are swallowed)."""
    try:
        os.makedirs(config_dir(), exist_ok=True)
        params_in = cfg.get("params") or {}
        clean = {
            "theme": cfg.get("theme") if cfg.get("theme") in VALID_THEMES else "light",
            "last_model": cfg.get("last_model") if isinstance(cfg.get("last_model"), str) else "",
            "params": {
                k: (params_in[k] if isinstance(params_in.get(k), (int, float))
                    else DEFAULT_PARAMS[k])
                for k in DEFAULT_PARAMS
            },
            "custom_prompts": {
                str(k): str(v) for k, v in (cfg.get("custom_prompts") or {}).items()
                if isinstance(v, str)
            },
        }
        tmp = config_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=2)
        os.replace(tmp, config_path())
    except Exception:
        pass


def get_theme():
    return load().get("theme", "light")


def set_theme(theme):
    if theme not in VALID_THEMES:
        return
    cfg = load()
    cfg["theme"] = theme
    save(cfg)


def get_last_model():
    return load().get("last_model", "")


def set_last_model(name):
    cfg = load()
    cfg["last_model"] = name or ""
    save(cfg)


def get_params():
    return load().get("params", dict(DEFAULT_PARAMS))


def set_params(params):
    cfg = load()
    merged = dict(DEFAULT_PARAMS)
    for k in DEFAULT_PARAMS:
        if isinstance((params or {}).get(k), (int, float)):
            merged[k] = params[k]
    cfg["params"] = merged
    save(cfg)


def get_custom_prompts():
    return load().get("custom_prompts", {})


def set_custom_prompt(name, text):
    cfg = load()
    prompts = dict(cfg.get("custom_prompts") or {})
    prompts[str(name)] = str(text)
    cfg["custom_prompts"] = prompts
    save(cfg)


def delete_custom_prompt(name):
    cfg = load()
    prompts = dict(cfg.get("custom_prompts") or {})
    prompts.pop(str(name), None)
    cfg["custom_prompts"] = prompts
    save(cfg)

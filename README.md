# LocalLLM Studio

A fast, **offline**, **100% open-source** local LLM chat studio for Windows. Nothing is uploaded anywhere. Built entirely by AI with human testing and guidance, and published on [QuickOpen](https://quickopen.ai/projects/localllm-studio).

> **100% AI-built and open source.** Apache-2.0.

## What it does

Run open-weight language models (GGUF) fully offline on your own hardware via llama.cpp: a chat interface with system prompts, conversation history, adjustable sampling, and simple retrieval over your own files. You supply the model file — nothing is sent to any server.

## Install

Download **`LocalLLMStudio-Setup.exe`** from the [QuickOpen page](https://quickopen.ai/projects/localllm-studio) or the [GitHub release](https://github.com/quickpod/localllm-studio/releases/latest) and double-click it. It installs per-user, adds Desktop and Start Menu shortcuts, and can optionally trust the QuickOpen Root CA. Authenticode-signed by the QuickOpen Code Signing CA — verify at [quickopen.ai/trust](https://quickopen.ai/trust).

## Run from source

```sh
pip install -r requirements.txt
python localllm_studio_app.py          # GUI
python -m localllm --help    # CLI
```

## Features

- **Chat, fully offline.** Pick an installed GGUF model, choose a system-prompt preset, and chat. Replies **stream** in token-by-token and generation runs **off the UI thread**, so the window never freezes — press **Stop** any time.
- **System prompts & presets.** Built-in presets (Assistant, Coding, Creative, Concise, Socratic) plus your own custom presets, saved to your config.
- **Parameter presets.** `assistant` / `coding` / `creative` bundle sensible sampling defaults; tweak temperature, top-p, top-k, max tokens, repeat penalty and seed with sliders that persist automatically.
- **Conversation management.** New chat, token-budget history trimming, and **Save/Load** a chat as JSON or export to Markdown.
- **Model manager.** See what's installed, open the models folder, and **download** small, permissively licensed GGUF models from a built-in catalog with a progress bar. Network access happens *only* when you explicitly download.
- **CLI + GUI** from one entry point. Dark/light themes (QuickOpen palette). Nothing is ever uploaded anywhere.

The native inference library (`llama-cpp-python`) is imported **lazily** — the app, its CLI and every menu open fine even before it's installed; you only need it to actually generate text.

## CLI examples

```sh
python -m localllm --version

# Models
python -m localllm models list                 # installed models + models dir
python -m localllm models catalog              # what you can download
python -m localllm models download tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# One-shot completion (add --stream to stream tokens)
python -m localllm complete --model tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --prompt "Write a haiku about offline AI." --max-tokens 64 --stream

# Interactive chat REPL: /reset  /system <text>  /save <file.json|.md>  /exit
python -m localllm chat --model tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --preset coding --temp 0.2
```

Every command exits cleanly with a friendly message when the model or the native library is missing (no traceback).

## Models (bring your own GGUF / download in-app)

LocalLLM Studio bundles no model weights — it runs any GGUF model you provide through **llama.cpp** (MIT). This keeps the app permissively licensed and lets you choose a model whose license suits you.

- **Bring your own:** drop any `*.gguf` file into the models folder — `%LOCALAPPDATA%\LocalLLMStudio\models` on Windows (`~/.localllmstudio/models` elsewhere) — or pass a full path to `--model`.
- **Download in-app:** the **Models** tab (and `localllm models download <name>`) fetches small, permissively licensed models from a curated catalog (TinyLlama 1.1B, Qwen2.5 0.5B/1.5B, Llama-3.2 1B). Downloads are always explicit and user-triggered.

## License

Apache-2.0 — see [LICENSE](LICENSE). A 100% AI-built project published on QuickOpen.

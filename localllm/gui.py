#!/usr/bin/env python3
r"""LocalLLM Studio -- an Aura (QuickOpen design system) GUI on the ``localllm`` API.

A single Aura window: sidebar sections **Chat**, **Parameters**, **Models** and
**Prompts**.  Text generation always runs on a background thread and streams
tokens back into the last chat bubble via ``self.after`` so the UI stays
responsive; failures surface in the Aura status bar as the ``LocalLLMError``
message (never a raw traceback).

Design goals baked in here (mirrors the QuickOpen house style):
  * built on the vendored ``localllm/aura.py`` design system, which layers the
    quickopen.ai look (deep space + light) over CustomTkinter.  Runtime deps:
    ``customtkinter`` (+ ``darkdetect``) — declared in requirements.txt; the
    PyInstaller build adds ``--collect-all customtkinter``.
  * Importing this module does nothing.  Only :func:`main` builds a root
    window, and it degrades gracefully (prints a message, returns 0) with no
    display or with customtkinter missing.
  * The native llama.cpp library is imported lazily by the engine, never here --
    so the GUI opens fine with no model and no native library installed; you
    only hit a friendly error when you actually press Send.
  * Frozen-exe safe: bundled assets are resolved via ``sys._MEIPASS`` / the exe
    directory when ``sys.frozen`` is set -- never ``__file__``.

100% AI-built, open source, published on QuickOpen (quickopen.ai).
"""

from __future__ import annotations

import os
import sys
import threading

# NOTE: tkinter/customtkinter are imported lazily inside main()/build_app so
# that merely importing this module (e.g. during packaging or on a headless CI
# box) never fails.

APP_NAME = "LocalLLM Studio"
APP_VERSION = "1.0.0"
WINDOW_TITLE = "LocalLLM Studio — by QuickOpen (quickopen.ai)"
PROJECT_URL = "https://quickopen.ai"
ACCENT = "#2f5fe0"      # publish/specs/localllm-studio.json "accent": [47, 95, 224]


# ---------------------------------------------------------------------------
# Asset / frozen handling
# ---------------------------------------------------------------------------
def asset_path(name):
    """Locate a bundled asset from source OR a PyInstaller one-file build.

    For a frozen exe we look only at ``sys._MEIPASS`` and the executable's own
    directory (never ``__file__``).  From source we also consult the package
    dir, the repo root and the CWD.  Returns an absolute path or ``None``.
    """
    roots = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(meipass)
        roots.append(os.path.dirname(os.path.abspath(sys.executable)))
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        roots += [here, os.path.dirname(here), os.getcwd()]
    for root in roots:
        candidate = os.path.join(root, name)
        if os.path.exists(candidate):
            return candidate
    return None


def open_with_default_app(path):
    """Open a file/URL with the OS default application, guarded."""
    try:
        if hasattr(os, "startfile"):
            os.startfile(path)                # noqa: S606
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", path])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        return False


def open_in_file_manager(path):
    """Best-effort 'reveal in file manager', guarded on every platform."""
    try:
        folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        return open_with_default_app(folder)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# The app (built lazily; tkinter/customtkinter imported only inside build_app)
# ---------------------------------------------------------------------------
def build_app():
    """Construct and return the App class bound to live GUI imports.

    Kept inside a function so this module imports cleanly without a display
    (and without customtkinter installed).
    """
    import tkinter as tk
    from tkinter import filedialog
    from .aura import filedialog  # noqa: F811 - Aura kdialog-native pickers
    import customtkinter as ctk

    from . import aura, guiconfig
    from . import prompts as prompts_mod
    from . import models as models_mod
    from .chatsession import ChatSession, preset_names
    from .engine import LlamaEngine, library_available
    from .errors import LocalLLMError

    PARAM_SPECS = [
        # (key, label, from, to, resolution, is_int)
        ("temperature", "Temperature", 0.0, 2.0, 0.05, False),
        ("top_p", "Top-p", 0.0, 1.0, 0.01, False),
        ("top_k", "Top-k", 0, 200, 1, True),
        ("max_tokens", "Max tokens", 16, 4096, 16, True),
        ("repeat_penalty", "Repeat penalty", 1.0, 2.0, 0.01, False),
        ("seed", "Seed (-1 = random)", -1, 999999, 1, True),
    ]

    class App(aura.AuraApp):
        def __init__(self):
            super().__init__(
                title=WINDOW_TITLE, app_name=APP_NAME, accent=ACCENT,
                theme=guiconfig.get_theme(),
                icon_png=asset_path("localllm-studio.png"),
                version=APP_VERSION, tagline="offline LLM chat",
                on_theme_change=guiconfig.set_theme,
                size=(1080, 720), min_size=(900, 600))

            # generation state
            self._engine = None
            self._engine_key = None     # (model, n_ctx) the engine was built for
            self._generating = False
            self._stop_flag = False
            self._img_refs_gui = []

            # persisted params + session
            self.params = dict(guiconfig.get_params())
            self.session = ChatSession(
                system=prompts_mod.default_prompt(),
                params=dict(self.params),
                max_context_tokens=4096,
            )

            self._set_icon()
            self._build_menu()
            self.add_section("chat", "Chat", "✉", self._panel_chat)
            self.add_section("params", "Parameters", "⚙", self._panel_params)
            self.add_section("models", "Models", "⛁", self._panel_models)
            self.add_section("prompts", "Prompts", "✎", self._panel_prompts)
            self.show("chat")
            self.set_status("Ready")
            self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- assets / icon
        def _set_icon(self):
            try:
                ico = asset_path("localllm-studio.ico")
                if ico and os.name == "nt":
                    self.iconbitmap(ico)
                    return
            except Exception:
                pass
            try:
                png = asset_path("localllm-studio.png")
                if png:
                    img = tk.PhotoImage(file=png)
                    self._img_refs_gui.append(img)
                    self.iconphoto(True, img)
            except Exception:
                pass  # icon is cosmetic; never block launch

        # ---- theming (bubble tags are raw-tk; re-tint on theme flip)
        def set_theme(self, theme):
            super().set_theme(theme)
            self._retint_transcript()

        def _retint_transcript(self):
            if not hasattr(self, "transcript"):
                return
            p = aura.P()
            try:
                self.transcript.tag_configure(
                    "user", foreground=p["accent"],
                    font=aura.font(10, "bold"), spacing1=10)
                self.transcript.tag_configure(
                    "bot", foreground=p["ok"],
                    font=aura.font(10, "bold"), spacing1=10)
                # chat bubbles: user = accent-soft, assistant = surface2
                self.transcript.tag_configure(
                    "user_body", background=p["accent_soft"],
                    foreground=p["text"], lmargin1=10, lmargin2=10,
                    rmargin=10, spacing1=4, spacing3=6)
                self.transcript.tag_configure(
                    "bot_body", background=p["surface2"],
                    foreground=p["text"], lmargin1=10, lmargin2=10,
                    rmargin=10, spacing1=4, spacing3=6)
            except Exception:
                pass

        # ---- menu (native menus stay)
        def _build_menu(self):
            bar = tk.Menu(self)
            filem = tk.Menu(bar, tearoff=0)
            filem.add_command(label="New chat", command=self._new_chat)
            filem.add_command(label="Save chat…", command=self._save_chat)
            filem.add_command(label="Load chat…", command=self._load_chat)
            filem.add_separator()
            filem.add_command(label="Open models folder",
                              command=lambda: open_in_file_manager(
                                  models_mod.models_dir(create=True)))
            filem.add_separator()
            filem.add_command(label="Exit", command=self._on_close)
            bar.add_cascade(label="File", menu=filem)

            viewm = tk.Menu(bar, tearoff=0)
            viewm.add_command(
                label="Toggle dark mode",
                command=lambda: self.set_theme(
                    "light" if self.theme == "dark" else "dark"))
            bar.add_cascade(label="View", menu=viewm)

            helpm = tk.Menu(bar, tearoff=0)
            helpm.add_command(label="About", command=self._about)
            helpm.add_command(label="Open project page (quickopen.ai)",
                              command=lambda: open_with_default_app(PROJECT_URL))
            bar.add_cascade(label="Help", menu=helpm)
            self.configure(menu=bar)

        # ---- section switching: refresh dynamic content on show
        def show(self, sid):
            super().show(sid)
            if sid == "chat":
                self._refresh_models_dropdown()
            elif sid == "models":
                self._refresh_installed_list()
            elif sid == "prompts":
                self._refresh_prompt_list()

        # ---- small helpers
        @staticmethod
        def _fill(entry, text):
            entry.delete(0, "end")
            if text:
                entry.insert(0, text)

        # =================================================================
        # CHAT
        # =================================================================
        def _panel_chat(self, parent):
            aura.Caption(
                parent, "Pick an installed model, choose a preset, and chat. "
                        "Generation streams in and runs off the UI thread; "
                        "press Stop any time.").pack(anchor="w", pady=(0, 10))

            top = ctk.CTkFrame(parent, fg_color="transparent")
            top.pack(fill="x")
            ctk.CTkLabel(top, text="Model", width=52, anchor="w",
                         font=aura.font()).pack(side="left")
            self.model_var = tk.StringVar(value=guiconfig.get_last_model())
            self.model_combo = aura.AuraCombo(
                top, variable=self.model_var, values=[""], state="readonly",
                width=280)
            self.model_combo.pack(side="left", padx=(0, 8))
            aura.AuraButton(top, "Refresh", kind="ghost",
                            command=self._refresh_models_dropdown).pack(side="left")
            ctk.CTkLabel(top, text="Preset", anchor="w",
                         font=aura.font()).pack(side="left", padx=(16, 6))
            self.preset_var = tk.StringVar(value="assistant")
            aura.AuraCombo(top, variable=self.preset_var, state="readonly",
                           width=140, values=preset_names(),
                           command=lambda _v: self._apply_preset()).pack(side="left")

            sysrow = ctk.CTkFrame(parent, fg_color="transparent")
            sysrow.pack(fill="x", pady=(10, 4))
            ctk.CTkLabel(sysrow, text="System", width=52, anchor="w",
                         font=aura.font()).pack(side="left")
            # no textvariable: CTkEntry placeholders only work without one
            self.system_entry = aura.AuraEntry(
                sysrow, placeholder="System prompt sent with every message…")
            self.system_entry.pack(side="left", fill="x", expand=True)
            self._fill(self.system_entry, self.session.system)
            self.system_entry.bind(
                "<KeyRelease>",
                lambda _e: self.session.set_system(self.system_entry.get()))

            # transcript (raw tk.Text: keeps exact streaming behaviour; themed
            # via aura.track + bubble tags re-tinted in set_theme)
            body = ctk.CTkFrame(parent, fg_color="transparent")
            body.pack(fill="both", expand=True, pady=8)
            self.transcript = tk.Text(body, wrap="word", width=20, height=12,
                                      state="disabled", font=aura.font(),
                                      relief="flat", padx=10, pady=8)
            sb = aura.AuraScrollbar(body, command=self.transcript.yview)
            self.transcript.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            self.transcript.pack(side="left", fill="both", expand=True)
            aura.track(self.transcript, "text")
            self._retint_transcript()

            # input row
            inrow = ctk.CTkFrame(parent, fg_color="transparent")
            inrow.pack(fill="x", pady=(4, 0))
            self.input_entry = aura.AuraEntry(
                inrow, placeholder="Type a message and press Enter…")
            self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self.input_entry.bind("<Return>", lambda e: self._send())
            self.send_btn = aura.AuraButton(inrow, "Send", command=self._send)
            self.send_btn.pack(side="left")
            self.stop_btn = aura.AuraButton(inrow, "Stop", kind="secondary",
                                            command=self._stop)
            self.stop_btn.pack(side="left", padx=(8, 0))
            self.stop_btn.state(["disabled"])

            ctrl = ctk.CTkFrame(parent, fg_color="transparent")
            ctrl.pack(fill="x", pady=(10, 0))
            aura.AuraButton(ctrl, "New chat", kind="secondary",
                            command=self._new_chat).pack(side="left")
            aura.AuraButton(ctrl, "Save…", kind="ghost",
                            command=self._save_chat).pack(side="left", padx=8)
            aura.AuraButton(ctrl, "Load…", kind="ghost",
                            command=self._load_chat).pack(side="left")

            if not library_available():
                aura.Caption(
                    parent,
                    "Note: the native llama.cpp library isn't installed, so "
                    "generation is disabled. Everything else works; run "
                    "`pip install llama-cpp-python` to enable chat.",
                    wraplength=720, justify="left").pack(anchor="w", pady=(8, 0))

        def _refresh_models_dropdown(self):
            if not hasattr(self, "model_combo"):
                return
            names = [m.name for m in models_mod.list_models()]
            self.model_combo.configure(values=names or [""])
            cur = self.model_var.get()
            if cur not in names:
                self.model_var.set(names[0] if names else "")

        def _apply_preset(self):
            try:
                self.session.apply_preset(self.preset_var.get(),
                                          prompts_module=prompts_mod)
            except LocalLLMError as exc:
                self.set_error(str(exc))
                return
            self._fill(self.system_entry, self.session.system)
            self.params.update(self.session.params)
            guiconfig.set_params(self.params)
            self._sync_param_widgets()

        def _append_transcript(self, who_tag, who_label, text):
            self.transcript.configure(state="normal")
            if self.transcript.index("end-1c") != "1.0":
                self.transcript.insert("end", "\n")
            self.transcript.insert("end", who_label + "\n", who_tag)
            self.transcript.insert("end", text, who_tag + "_body")
            self.transcript.configure(state="disabled")
            self.transcript.see("end")

        def _append_stream(self, piece):
            self.transcript.configure(state="normal")
            self.transcript.insert("end", piece, "bot_body")
            self.transcript.configure(state="disabled")
            self.transcript.see("end")

        def _ensure_engine(self, model_name):
            key = (model_name, 4096)
            if self._engine is not None and self._engine_key == key:
                return self._engine
            self._engine = LlamaEngine(model_name, n_ctx=4096)
            self._engine_key = key
            return self._engine

        def _send(self):
            if self._generating:
                return
            model_name = self.model_var.get().strip()
            text = self.input_entry.get().strip()
            if not model_name:
                self.set_error("Choose a model first (Models tab to download one).")
                return
            if not text:
                return
            self._fill(self.input_entry, "")
            self.session.set_system(self.system_entry.get())
            self.session.params = dict(self.params)
            self.session.add_user(text)
            self._append_transcript("user", "You", text)
            self._append_transcript("bot", "Assistant", "")

            self._generating = True
            self._stop_flag = False
            self.send_btn.state(["disabled"])
            self.stop_btn.state(["!disabled"])
            self.set_status("Generating…", kind="working")

            messages = self.session.build_messages()
            params = dict(self.session.params)
            collected = []

            def worker():
                try:
                    eng = self._ensure_engine(model_name)
                    for piece in eng.stream_chat(messages, **params):
                        if self._stop_flag:
                            break
                        collected.append(piece)
                        self.after(0, lambda p=piece: self._append_stream(p))
                    self.after(0, lambda: finish(None))
                except LocalLLMError as exc:
                    self.after(0, lambda e=exc: finish(str(e)))
                except Exception as exc:  # never leak a traceback
                    self.after(0, lambda e=exc: finish(f"Unexpected error: {e}"))

            def finish(err):
                self._generating = False
                self.send_btn.state(["!disabled"])
                self.stop_btn.state(["disabled"])
                reply = "".join(collected)
                self.session.add_assistant(reply)
                if err:
                    self.after(0, lambda: self._append_stream(""))
                    self.set_error(err)
                elif self._stop_flag:
                    self.set_status("Stopped.", kind="idle")
                else:
                    self.set_success("Done.")

            threading.Thread(target=worker, daemon=True).start()

        def _stop(self):
            self._stop_flag = True

        def _new_chat(self):
            self.session = ChatSession(
                system=self.system_entry.get() if hasattr(self, "system_entry")
                else prompts_mod.default_prompt(),
                params=dict(self.params), max_context_tokens=4096)
            if hasattr(self, "transcript"):
                self.transcript.configure(state="normal")
                self.transcript.delete("1.0", "end")
                self.transcript.configure(state="disabled")
            self.set_status("Ready")

        def _save_chat(self):
            path = filedialog.asksaveasfilename(
                title="Save chat", defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("Markdown", "*.md"),
                           ("All files", "*.*")])
            if not path:
                return
            try:
                self.session.model = self.model_var.get() if hasattr(self, "model_var") else ""
                self.session.save(path)
                self.set_success(f"Saved chat → {path}")
            except LocalLLMError as exc:
                self.set_error(str(exc))

        def _load_chat(self):
            path = filedialog.askopenfilename(
                title="Load chat (JSON)",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            if not path:
                return
            try:
                self.session = ChatSession.load(path, max_context_tokens=4096)
            except LocalLLMError as exc:
                self.set_error(str(exc))
                return
            if hasattr(self, "system_entry"):
                self._fill(self.system_entry, self.session.system)
            if hasattr(self, "transcript"):
                self.transcript.configure(state="normal")
                self.transcript.delete("1.0", "end")
                self.transcript.configure(state="disabled")
                for m in self.session.messages:
                    if m["role"] == "user":
                        self._append_transcript("user", "You", m["content"])
                    else:
                        self._append_transcript("bot", "Assistant", m["content"])
            self.set_success(f"Loaded chat ← {path}")

        # =================================================================
        # PARAMETERS
        # =================================================================
        def _panel_params(self, parent):
            aura.Caption(
                parent, "Sampling settings applied to every generation. "
                        "Saved automatically.").pack(anchor="w", pady=(0, 10))
            card = aura.Card(parent, title="Sampling")
            card.pack(fill="x")
            self._param_vars = {}
            self._param_labels = {}
            grid = ctk.CTkFrame(card.body, fg_color="transparent")
            grid.pack(fill="x")
            for i, (key, label, lo, hi, res, is_int) in enumerate(PARAM_SPECS):
                ctk.CTkLabel(grid, text=label, width=150, anchor="w",
                             font=aura.font()).grid(
                    row=i, column=0, sticky="w", pady=6)
                var = tk.DoubleVar(value=float(self.params.get(key, 0)))
                self._param_vars[key] = (var, is_int)
                steps = max(1, int(round((hi - lo) / res)))
                slider = ctk.CTkSlider(
                    grid, from_=lo, to=hi, variable=var,
                    number_of_steps=min(steps, 4096), height=16,
                    command=lambda v, k=key: self._on_param_change(k))
                slider.grid(row=i, column=1, sticky="we", padx=10)
                vlbl = ctk.CTkLabel(grid, text=self._fmt_param(key), width=70,
                                    anchor="e", font=aura.font())
                vlbl.grid(row=i, column=2, sticky="e")
                self._param_labels[key] = vlbl
            grid.columnconfigure(1, weight=1)

            btns = ctk.CTkFrame(card.body, fg_color="transparent")
            btns.pack(fill="x", pady=(12, 0))
            aura.AuraButton(btns, "Reset to defaults", kind="secondary",
                            command=self._reset_params).pack(side="left")
            aura.Caption(parent,
                         "Parameters are saved automatically and applied to "
                         "every generation.").pack(anchor="w", pady=(10, 0))

        def _fmt_param(self, key):
            for k, _l, _lo, _hi, _r, is_int in PARAM_SPECS:
                if k == key:
                    v = self.params.get(key, 0)
                    return str(int(round(v))) if is_int else f"{float(v):.2f}"
            return str(self.params.get(key, 0))

        def _on_param_change(self, key):
            var, is_int = self._param_vars[key]
            v = var.get()
            self.params[key] = int(round(v)) if is_int else round(float(v), 3)
            self._param_labels[key].configure(text=self._fmt_param(key))
            self.session.params = dict(self.params)
            guiconfig.set_params(self.params)

        def _sync_param_widgets(self):
            if not hasattr(self, "_param_vars"):
                return
            for key, (var, _is_int) in self._param_vars.items():
                var.set(float(self.params.get(key, 0)))
                self._param_labels[key].configure(text=self._fmt_param(key))

        def _reset_params(self):
            self.params = dict(guiconfig.DEFAULT_PARAMS)
            guiconfig.set_params(self.params)
            self.session.params = dict(self.params)
            self._sync_param_widgets()
            self.set_success("Parameters reset to defaults.")

        # =================================================================
        # MODELS
        # =================================================================
        def _panel_models(self, parent):
            aura.Caption(
                parent, "Installed GGUF models and one-click downloads of "
                        "small, permissively licensed models.").pack(
                anchor="w", pady=(0, 10))

            inst = aura.Card(parent, title="Installed")
            inst.pack(fill="both", expand=True)
            dirrow = ctk.CTkFrame(inst.body, fg_color="transparent")
            dirrow.pack(fill="x", pady=(0, 8))
            self.modeldir_lbl = aura.Caption(
                dirrow, "Models folder: " + models_mod.models_dir())
            self.modeldir_lbl.pack(side="left")
            aura.AuraButton(dirrow, "Open folder", kind="ghost",
                            command=lambda: open_in_file_manager(
                                models_mod.models_dir(create=True))).pack(
                side="right")
            self.installed_list = tk.Listbox(inst.body, height=6,
                                             activestyle="none",
                                             exportselection=False,
                                             font=aura.font())
            self.installed_list.pack(fill="both", expand=True)
            aura.track(self.installed_list, "listbox")
            aura.AuraButton(inst.body, "Refresh", kind="secondary",
                            command=self._refresh_installed_list).pack(
                anchor="w", pady=(10, 0))

            dl = aura.Card(parent, title="Download from catalog")
            dl.pack(fill="x", pady=(14, 0))
            crow = ctk.CTkFrame(dl.body, fg_color="transparent")
            crow.pack(fill="x")
            self.catalog_var = tk.StringVar()
            names = [e.name for e in models_mod.catalog()]
            self.catalog_combo = aura.AuraCombo(
                crow, variable=self.catalog_var, state="readonly", width=380,
                values=names or [""],
                command=lambda _v: self._update_catalog_note())
            if names:
                self.catalog_var.set(names[0])
            self.catalog_combo.pack(side="left", padx=(0, 10))
            self.dl_btn = aura.AuraButton(crow, "Download",
                                          command=self._download_model)
            self.dl_btn.pack(side="left")
            self.dl_progress = aura.ProgressBar(dl.body)   # 0..1 scale
            self.dl_progress.pack(fill="x", pady=(12, 4))
            self.dl_progress.set(0)
            self.catalog_note = aura.Caption(dl.body, "", wraplength=720,
                                             justify="left")
            self.catalog_note.pack(anchor="w")
            self._update_catalog_note()

        def _refresh_installed_list(self):
            if not hasattr(self, "installed_list"):
                return
            self.installed_list.delete(0, "end")
            installed = models_mod.list_models()
            if not installed:
                self.installed_list.insert("end", "(no models installed yet)")
            for m in installed:
                self.installed_list.insert(
                    "end", f"{m.name}   ({models_mod.human_size(m.size)})")

        def _update_catalog_note(self):
            e = models_mod.find_catalog_entry(self.catalog_var.get())
            if e:
                self.catalog_note.configure(
                    text=f"{e.params} params · {e.quant} · "
                         f"~{models_mod.human_size(e.approx_size)} · {e.license}\n"
                         f"{e.note}")

        def _download_model(self):
            name = self.catalog_var.get().strip()
            if not name:
                return
            self.dl_btn.state(["disabled"])
            self.dl_progress.set(0)
            self.set_status("Downloading…", kind="working")

            def progress(done, total):
                frac = done / total if total else 0.0
                self.after(0, lambda: self.dl_progress.set(min(1.0, frac)))

            def worker():
                try:
                    dest = models_mod.download(name, progress=progress)
                    self.after(0, lambda: finish(dest, None))
                except LocalLLMError as exc:
                    self.after(0, lambda e=exc: finish(None, str(e)))
                except Exception as exc:
                    self.after(0, lambda e=exc: finish(None, f"Unexpected error: {e}"))

            def finish(dest, err):
                self.dl_btn.state(["!disabled"])
                if err:
                    self.set_error(err)
                    return
                self.dl_progress.set(1.0)
                self.set_success(f"Downloaded → {dest}")
                self._refresh_installed_list()
                self._refresh_models_dropdown()

            threading.Thread(target=worker, daemon=True).start()

        # =================================================================
        # PROMPTS
        # =================================================================
        def _panel_prompts(self, parent):
            aura.Caption(
                parent, "Built-in and custom system-prompt presets.").pack(
                anchor="w", pady=(0, 10))
            body = ctk.CTkFrame(parent, fg_color="transparent")
            body.pack(fill="both", expand=True)
            # fixed column packed BEFORE the expanding one (pack shrinks later
            # children first)
            left = aura.Card(body, title="Presets")
            left.pack(side="left", fill="y")
            self.prompt_list = tk.Listbox(left.body, height=12,
                                          activestyle="none",
                                          exportselection=False, width=24,
                                          font=aura.font())
            self.prompt_list.pack(side="left", fill="y", expand=True)
            self.prompt_list.bind("<<ListboxSelect>>",
                                  lambda e: self._on_prompt_select())
            aura.track(self.prompt_list, "listbox")

            right = aura.Card(body, title="Edit preset")
            right.pack(side="left", fill="both", expand=True, padx=(14, 0))
            self.prompt_name_entry = aura.AuraEntry(
                right.body, placeholder="Preset name")
            self.prompt_name_entry.pack(fill="x")
            aura.Caption(right.body, "System prompt").pack(
                anchor="w", pady=(10, 2))
            self.prompt_text = tk.Text(right.body, wrap="word", width=20,
                                       height=10, font=aura.font(),
                                       relief="flat", padx=8, pady=6)
            self.prompt_text.pack(fill="both", expand=True)
            aura.track(self.prompt_text, "text")

            btns = ctk.CTkFrame(right.body, fg_color="transparent")
            btns.pack(fill="x", pady=(10, 0))
            aura.AuraButton(btns, "Save preset",
                            command=self._save_prompt).pack(side="left")
            aura.AuraButton(btns, "Delete", kind="danger",
                            command=self._delete_prompt).pack(
                side="left", padx=8)
            aura.AuraButton(btns, "Use in chat", kind="secondary",
                            command=self._use_prompt).pack(side="left")

        def _refresh_prompt_list(self):
            if not hasattr(self, "prompt_list"):
                return
            self.prompt_list.delete(0, "end")
            for n in prompts_mod.builtin_names():
                self.prompt_list.insert("end", n)
            customs = prompts_mod.custom_names()
            if customs:
                self.prompt_list.insert("end", "— custom —")
                for n in customs:
                    self.prompt_list.insert("end", n)

        def _on_prompt_select(self):
            sel = self.prompt_list.curselection()
            if not sel:
                return
            name = self.prompt_list.get(sel[0])
            if name.startswith("—"):
                return
            try:
                text = prompts_mod.get_prompt(name)
            except LocalLLMError:
                return
            self._fill(self.prompt_name_entry, name)
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", text)

        def _save_prompt(self):
            name = self.prompt_name_entry.get().strip()
            text = self.prompt_text.get("1.0", "end").strip()
            try:
                prompts_mod.save_custom(name, text)
            except LocalLLMError as exc:
                self.set_error(str(exc))
                return
            self._refresh_prompt_list()
            self.set_success(f"Saved preset '{name}'.")

        def _delete_prompt(self):
            name = self.prompt_name_entry.get().strip()
            try:
                prompts_mod.delete_custom(name)
            except LocalLLMError as exc:
                self.set_error(str(exc))
                return
            self._refresh_prompt_list()
            self.set_success(f"Deleted preset '{name}'.")

        def _use_prompt(self):
            text = self.prompt_text.get("1.0", "end").strip()
            self.session.set_system(text)
            if hasattr(self, "system_entry"):
                self._fill(self.system_entry, text)
            self.set_success("Applied to the current chat's system prompt.")

        # ---- About
        def _about(self):
            win = ctk.CTkToplevel(self)
            win.title("About LocalLLM Studio")
            win.resizable(False, False)
            frm = ctk.CTkFrame(win, fg_color="transparent")
            frm.pack(fill="both", expand=True, padx=20, pady=18)
            aura.Heading(frm, APP_NAME).pack(anchor="w")
            aura.Caption(frm, f"Version {APP_VERSION}").pack(
                anchor="w", pady=(0, 10))
            ctk.CTkLabel(
                frm, font=aura.font(), justify="left", anchor="w",
                wraplength=400,
                text="Run open-weight GGUF language models fully offline on "
                     "your own hardware via llama.cpp.\n\n"
                     "100% AI-built, open source, published on QuickOpen.\n"
                     "Nothing is ever uploaded anywhere.").pack(anchor="w")
            aura.Caption(
                frm, "Licensed under Apache-2.0. Built on llama-cpp-python "
                     "(MIT) and CustomTkinter (MIT). You bring your own "
                     "model file.", wraplength=400, justify="left").pack(
                anchor="w", pady=(8, 10))
            aura.AuraButton(frm, "Close", kind="secondary",
                            command=win.destroy).pack(anchor="e")
            win.transient(self)
            win.grab_set()

        # ---- close
        def _on_close(self):
            try:
                guiconfig.set_params(self.params)
                if hasattr(self, "model_var"):
                    guiconfig.set_last_model(self.model_var.get())
            except Exception:
                pass
            self.destroy()

    return App


def main():
    """Launch the GUI. Side-effect-free to import; headless-guarded.

    Returns 0 on a clean exit (including when there is no display or when
    customtkinter is missing, where it prints a short message and returns
    instead of raising).
    """
    try:
        import tkinter as tk  # noqa: F401 - just probing availability
    except Exception as exc:
        sys.stderr.write(f"{APP_NAME}: tkinter is not available ({exc}).\n")
        return 0

    try:
        App = build_app()
        app = App()
    except ImportError as exc:
        sys.stderr.write(
            f"{APP_NAME}: the GUI needs the 'customtkinter' package ({exc}). "
            f"Install it with:  pip install customtkinter\n")
        return 0
    except Exception as exc:
        name = type(exc).__name__
        if "Tcl" in name or "display" in str(exc).lower() or "DISPLAY" in str(exc):
            sys.stderr.write(
                f"{APP_NAME}: no display available; GUI not started.\n")
            return 0
        raise
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

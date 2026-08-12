#!/usr/bin/env python3
r"""LocalLLM Studio -- a pure-stdlib tkinter GUI on top of the ``localllm`` API.

A single main window: a left sidebar (Chat, Parameters, Models, Prompts) and a
main panel that swaps to the selected section.  Text generation always runs on a
background thread and streams tokens back into the last chat bubble via
``self.after`` so the UI stays responsive; failures surface inline as the
``LocalLLMError`` message (never a raw traceback).

Design goals baked in here (mirrors the sibling PDF Toolkit app):
  * pure standard-library tkinter/ttk -- NO third-party GUI deps.  Dark mode is
    a ttk-style + palette swap.
  * Importing this module does nothing.  Only :func:`main` builds a root window,
    and it degrades gracefully (prints a message, returns 0) with no display.
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

# NOTE: tkinter is imported lazily inside main()/build_app so that merely
# importing this module (e.g. during packaging or on a headless CI box) never
# fails.

APP_NAME = "LocalLLM Studio"
APP_VERSION = "1.0.0"
WINDOW_TITLE = "LocalLLM Studio — by QuickOpen (quickopen.ai)"
PROJECT_URL = "https://quickopen.ai"

# (section_id, label) for the sidebar.
SECTIONS = [
    ("chat", "Chat"),
    ("params", "Parameters"),
    ("models", "Models"),
    ("prompts", "Prompts"),
]

# ---- colour palettes (mirror the QuickOpen palette) -------------------------
PALETTES = {
    "light": {
        "bg": "#f5f7fa", "surface": "#ffffff", "text": "#141820",
        "muted": "#5b6472", "primary": "#2f5fe0", "primary_hi": "#2450c8",
        "entry": "#ffffff", "border": "#d5dae2", "sel": "#2f5fe0",
        "sel_fg": "#ffffff", "trough": "#e2e7ef", "ok": "#1f7a3d",
        "err": "#c0392b", "bubble_user": "#e6edff", "bubble_bot": "#eef1f5",
    },
    "dark": {
        "bg": "#0f1115", "surface": "#1a1e24", "text": "#f1f3f7",
        "muted": "#9aa4b2", "primary": "#5b86f7", "primary_hi": "#7098ff",
        "entry": "#1a1e24", "border": "#2a2f38", "sel": "#5b86f7",
        "sel_fg": "#0f1115", "trough": "#2a2f38", "ok": "#5bd68a",
        "err": "#ff6b5e", "bubble_user": "#22304d", "bubble_bot": "#20262f",
    },
}


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
# The app (built lazily; tkinter imported only inside build_app/main)
# ---------------------------------------------------------------------------
def build_app():
    """Construct and return the App class bound to a live tkinter import.

    Kept inside a function so this module imports cleanly without a display.
    """
    import tkinter as tk
    from tkinter import ttk, filedialog

    from . import guiconfig
    from . import prompts as prompts_mod
    from . import models as models_mod
    from .chatsession import ChatSession, preset_names
    from .engine import LlamaEngine, DEFAULT_PARAMS, library_available
    from .errors import LocalLLMError

    FONT = "Segoe UI"

    PARAM_SPECS = [
        # (key, label, from, to, resolution, is_int)
        ("temperature", "Temperature", 0.0, 2.0, 0.05, False),
        ("top_p", "Top-p", 0.0, 1.0, 0.01, False),
        ("top_k", "Top-k", 0, 200, 1, True),
        ("max_tokens", "Max tokens", 16, 4096, 16, True),
        ("repeat_penalty", "Repeat penalty", 1.0, 2.0, 0.01, False),
        ("seed", "Seed (-1 = random)", -1, 999999, 1, True),
    ]

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(WINDOW_TITLE)
            self.geometry("1080x720")
            self.minsize(900, 600)

            self.theme = guiconfig.get_theme()
            self._tracked = []          # (tk_widget, role) for manual re-theming
            self._img_refs = []
            self._panels = {}
            self._current = None

            # generation state
            self._engine = None
            self._engine_key = None     # (model, n_ctx) the engine was built for
            self._generating = False
            self._stop_flag = False

            # persisted params + session
            self.params = dict(guiconfig.get_params())
            self.session = ChatSession(
                system=prompts_mod.default_prompt(),
                params=dict(self.params),
                max_context_tokens=4096,
            )

            self._set_icon()
            self._build_menu()
            self._build_layout()
            self._apply_theme()
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            self.after(50, lambda: self._select_section("chat"))

        # ---- assets / icon
        def _set_icon(self):
            try:
                ico = asset_path("localllm-studio.ico")
                if ico:
                    self.iconbitmap(ico)
                    return
            except Exception:
                pass
            try:
                png = asset_path("localllm-studio.png")
                if png:
                    img = tk.PhotoImage(file=png)
                    self._img_refs.append(img)
                    self.iconphoto(True, img)
            except Exception:
                pass  # icon is cosmetic; never block launch

        # ---- theming
        def track(self, widget, role):
            self._tracked.append((widget, role))

        def _pal(self):
            return PALETTES[self.theme]

        def _apply_theme(self):
            p = self._pal()
            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            self.configure(bg=p["bg"])
            style.configure(".", background=p["bg"], foreground=p["text"],
                            fieldbackground=p["entry"], bordercolor=p["border"],
                            font=(FONT, 10))
            style.configure("TFrame", background=p["bg"])
            style.configure("Sidebar.TFrame", background=p["surface"])
            style.configure("Card.TFrame", background=p["surface"])
            style.configure("TLabel", background=p["bg"], foreground=p["text"])
            style.configure("Muted.TLabel", background=p["bg"], foreground=p["muted"])
            style.configure("Header.TLabel", background=p["bg"], foreground=p["text"],
                            font=(FONT, 15, "bold"))
            style.configure("Sub.TLabel", background=p["bg"], foreground=p["muted"],
                            font=(FONT, 10))
            style.configure("Brand.TLabel", background=p["surface"],
                            foreground=p["text"], font=(FONT, 12, "bold"))
            style.configure("Status.TLabel", background=p["surface"],
                            foreground=p["muted"])
            style.configure("TButton", background=p["surface"], foreground=p["text"],
                            bordercolor=p["border"], focuscolor=p["surface"],
                            padding=(10, 5))
            style.map("TButton",
                      background=[("active", p["trough"]), ("disabled", p["bg"])],
                      foreground=[("disabled", p["muted"])])
            style.configure("Accent.TButton", background=p["primary"],
                            foreground="#ffffff", padding=(12, 6))
            style.map("Accent.TButton",
                      background=[("active", p["primary_hi"]),
                                  ("disabled", p["border"])],
                      foreground=[("disabled", p["muted"])])
            style.configure("Toggle.TButton", background=p["surface"],
                            foreground=p["text"], padding=(8, 4))
            style.configure("Nav.TButton", background=p["surface"],
                            foreground=p["text"], anchor="w", padding=(14, 10))
            style.map("Nav.TButton",
                      background=[("active", p["trough"])])
            style.configure("NavSel.TButton", background=p["primary"],
                            foreground="#ffffff", anchor="w", padding=(14, 10))
            style.map("NavSel.TButton", background=[("active", p["primary_hi"])])
            for name in ("TEntry", "TSpinbox"):
                style.configure(name, fieldbackground=p["entry"], foreground=p["text"],
                                insertcolor=p["text"], bordercolor=p["border"])
            style.configure("TCombobox", fieldbackground=p["entry"],
                            foreground=p["text"], background=p["surface"],
                            arrowcolor=p["text"])
            style.map("TCombobox",
                      fieldbackground=[("readonly", p["entry"])],
                      foreground=[("readonly", p["text"])])
            style.configure("TScale", background=p["bg"], troughcolor=p["trough"])
            style.configure("Horizontal.TScale", background=p["bg"],
                            troughcolor=p["trough"])
            style.configure("Horizontal.TProgressbar", background=p["primary"],
                            troughcolor=p["trough"], bordercolor=p["border"])
            style.configure("TScrollbar", background=p["surface"],
                            troughcolor=p["bg"], bordercolor=p["border"],
                            arrowcolor=p["text"])
            style.configure("TSeparator", background=p["border"])
            style.configure("Treeview", background=p["surface"],
                            fieldbackground=p["surface"], foreground=p["text"],
                            bordercolor=p["border"], rowheight=24)
            style.map("Treeview", background=[("selected", p["primary"])],
                      foreground=[("selected", p["sel_fg"])])

            for widget, role in list(self._tracked):
                try:
                    if role == "listbox":
                        widget.configure(bg=p["surface"], fg=p["text"],
                                         selectbackground=p["primary"],
                                         selectforeground=p["sel_fg"],
                                         highlightthickness=1,
                                         highlightbackground=p["border"],
                                         borderwidth=0)
                    elif role == "text":
                        widget.configure(bg=p["surface"], fg=p["text"],
                                         insertbackground=p["text"],
                                         selectbackground=p["primary"],
                                         selectforeground=p["sel_fg"],
                                         highlightthickness=1,
                                         highlightbackground=p["border"],
                                         borderwidth=0)
                        try:
                            widget.tag_configure("user", foreground=p["primary"],
                                                 font=(FONT, 10, "bold"))
                            widget.tag_configure("bot", foreground=p["ok"],
                                                 font=(FONT, 10, "bold"))
                            widget.tag_configure("body", foreground=p["text"])
                        except Exception:
                            pass
                except Exception:
                    pass

        def toggle_theme(self):
            self.theme = "dark" if self.theme == "light" else "light"
            guiconfig.set_theme(self.theme)
            self._apply_theme()
            self._theme_btn.configure(
                text="☀ Light mode" if self.theme == "dark" else "🌙 Dark mode")

        # ---- menu
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
            viewm.add_command(label="Toggle dark mode", command=self.toggle_theme)
            bar.add_cascade(label="View", menu=viewm)

            helpm = tk.Menu(bar, tearoff=0)
            helpm.add_command(label="About", command=self._about)
            helpm.add_command(label="Open project page (quickopen.ai)",
                              command=lambda: open_with_default_app(PROJECT_URL))
            bar.add_cascade(label="Help", menu=helpm)
            self.configure(menu=bar)

        # ---- layout
        def _build_layout(self):
            top = ttk.Frame(self, style="Sidebar.TFrame", padding=(12, 8))
            top.pack(fill="x", side="top")
            ttk.Label(top, text="LocalLLM Studio", style="Brand.TLabel").pack(side="left")
            ttk.Label(top, style="Status.TLabel",
                      text="  offline · open source · by QuickOpen").pack(side="left")
            self._theme_btn = ttk.Button(
                top, style="Toggle.TButton", command=self.toggle_theme,
                text="☀ Light mode" if self.theme == "dark" else "🌙 Dark mode")
            self._theme_btn.pack(side="right")

            body = ttk.Frame(self, style="TFrame")
            body.pack(fill="both", expand=True)

            side = ttk.Frame(body, style="Sidebar.TFrame", width=200)
            side.pack(side="left", fill="y")
            side.pack_propagate(False)
            self._nav_btns = {}
            for sid, label in SECTIONS:
                b = ttk.Button(side, text=label, style="Nav.TButton",
                               command=lambda s=sid: self._select_section(s))
                b.pack(fill="x", padx=8, pady=(8 if sid == "chat" else 2, 0))
                self._nav_btns[sid] = b

            main = ttk.Frame(body, style="TFrame", padding=(16, 12))
            main.pack(side="left", fill="both", expand=True)

            head = ttk.Frame(main, style="TFrame")
            head.pack(fill="x")
            self.title_lbl = ttk.Label(head, text="Chat", style="Header.TLabel")
            self.title_lbl.pack(anchor="w")
            self.desc_lbl = ttk.Label(head, text="", style="Sub.TLabel",
                                      wraplength=760, justify="left")
            self.desc_lbl.pack(anchor="w", pady=(2, 8))
            ttk.Separator(main).pack(fill="x")

            self.container = ttk.Frame(main, style="TFrame")
            self.container.pack(fill="both", expand=True, pady=(10, 8))

            # shared inline status/result bar
            bar = ttk.Frame(self, style="Sidebar.TFrame", padding=(12, 6))
            bar.pack(fill="x", side="bottom")
            self.status_lbl = ttk.Label(bar, text="Ready", style="Status.TLabel",
                                        width=16, anchor="w")
            self.status_lbl.pack(side="left")
            self.result_lbl = ttk.Label(bar, text="", style="Status.TLabel",
                                        anchor="w", wraplength=760, justify="left")
            self.result_lbl.pack(side="left", fill="x", expand=True, padx=8)

        # ---- section switching
        def _select_section(self, sid):
            if self._current is not None:
                self._current.pack_forget()
            panel = self._panels.get(sid)
            if panel is None:
                panel = ttk.Frame(self.container, style="TFrame")
                builder = getattr(self, "_panel_" + sid, None)
                if builder:
                    builder(panel)
                self._panels[sid] = panel
                self._apply_theme()
            panel.pack(fill="both", expand=True)
            self._current = panel
            for s, b in self._nav_btns.items():
                b.configure(style="NavSel.TButton" if s == sid else "Nav.TButton")
            meta = {
                "chat": ("Chat", "Pick an installed model, choose a preset, and "
                                 "chat. Generation streams in and runs off the UI "
                                 "thread; press Stop any time."),
                "params": ("Parameters", "Sampling settings applied to every "
                                          "generation. Saved automatically."),
                "models": ("Models", "Installed GGUF models and one-click "
                                      "downloads of small, permissively licensed "
                                      "models."),
                "prompts": ("Prompts", "Built-in and custom system-prompt "
                                       "presets."),
            }.get(sid, (sid, ""))
            self.title_lbl.configure(text=meta[0])
            self.desc_lbl.configure(text=meta[1])
            self._clear_result()
            # refresh dynamic content
            if sid == "chat":
                self._refresh_models_dropdown()
            elif sid == "models":
                self._refresh_installed_list()
            elif sid == "prompts":
                self._refresh_prompt_list()

        # ---- status/result helpers
        def _set_status(self, text, kind="idle"):
            p = self._pal()
            color = {"working": p["primary"], "ok": p["ok"], "err": p["err"]}.get(
                kind, p["muted"])
            self.status_lbl.configure(text=text, foreground=color)

        def _clear_result(self):
            self.result_lbl.configure(text="")
            self._set_status("Ready")

        def _show_error(self, message):
            self.result_lbl.configure(text="✕ " + str(message),
                                      foreground=self._pal()["err"])
            self._set_status("error", kind="err")

        def _show_ok(self, message):
            self.result_lbl.configure(text="✓ " + str(message),
                                      foreground=self._pal()["ok"])
            self._set_status("done", kind="ok")

        # =================================================================
        # CHAT
        # =================================================================
        def _panel_chat(self, parent):
            top = ttk.Frame(parent, style="TFrame")
            top.pack(fill="x")
            ttk.Label(top, text="Model", width=8, anchor="w").pack(side="left")
            self.model_var = tk.StringVar(value=guiconfig.get_last_model())
            self.model_combo = ttk.Combobox(top, textvariable=self.model_var,
                                             state="readonly", width=36)
            self.model_combo.pack(side="left", padx=(0, 8))
            ttk.Button(top, text="Refresh",
                       command=self._refresh_models_dropdown).pack(side="left")
            ttk.Label(top, text="Preset").pack(side="left", padx=(16, 4))
            self.preset_var = tk.StringVar(value="assistant")
            pc = ttk.Combobox(top, textvariable=self.preset_var, state="readonly",
                              width=12, values=preset_names())
            pc.pack(side="left")
            pc.bind("<<ComboboxSelected>>", lambda e: self._apply_preset())

            sysrow = ttk.Frame(parent, style="TFrame")
            sysrow.pack(fill="x", pady=(8, 4))
            ttk.Label(sysrow, text="System", width=8, anchor="w").pack(side="left")
            self.system_var = tk.StringVar(value=self.session.system)
            sysent = ttk.Entry(sysrow, textvariable=self.system_var)
            sysent.pack(side="left", fill="x", expand=True)
            self.system_var.trace_add(
                "write", lambda *_: self.session.set_system(self.system_var.get()))

            # transcript
            body = ttk.Frame(parent, style="TFrame")
            body.pack(fill="both", expand=True, pady=6)
            self.transcript = tk.Text(body, wrap="word", height=16, state="disabled")
            sb = ttk.Scrollbar(body, orient="vertical", command=self.transcript.yview)
            self.transcript.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
            self.transcript.pack(side="left", fill="both", expand=True)
            self.track(self.transcript, "text")

            # input row
            inrow = ttk.Frame(parent, style="TFrame")
            inrow.pack(fill="x", pady=(4, 0))
            self.input_var = tk.StringVar()
            self.input_entry = ttk.Entry(inrow, textvariable=self.input_var)
            self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
            self.input_entry.bind("<Return>", lambda e: self._send())
            self.send_btn = ttk.Button(inrow, text="Send", style="Accent.TButton",
                                       command=self._send)
            self.send_btn.pack(side="left")
            self.stop_btn = ttk.Button(inrow, text="Stop", command=self._stop,
                                       state="disabled")
            self.stop_btn.pack(side="left", padx=(6, 0))

            ctrl = ttk.Frame(parent, style="TFrame")
            ctrl.pack(fill="x", pady=(6, 0))
            ttk.Button(ctrl, text="New chat", command=self._new_chat).pack(side="left")
            ttk.Button(ctrl, text="Save…", command=self._save_chat).pack(
                side="left", padx=6)
            ttk.Button(ctrl, text="Load…", command=self._load_chat).pack(side="left")

            if not library_available():
                ttk.Label(parent, style="Muted.TLabel", wraplength=760,
                          text="Note: the native llama.cpp library isn't installed, "
                               "so generation is disabled. Everything else works; "
                               "run `pip install llama-cpp-python` to enable chat."
                          ).pack(anchor="w", pady=(6, 0))

        def _refresh_models_dropdown(self):
            if not hasattr(self, "model_combo"):
                return
            names = [m.name for m in models_mod.list_models()]
            self.model_combo.configure(values=names)
            cur = self.model_var.get()
            if cur not in names:
                self.model_var.set(names[0] if names else "")

        def _apply_preset(self):
            try:
                self.session.apply_preset(self.preset_var.get(),
                                          prompts_module=prompts_mod)
            except LocalLLMError as exc:
                self._show_error(exc)
                return
            self.system_var.set(self.session.system)
            self.params.update(self.session.params)
            guiconfig.set_params(self.params)
            if "params" in self._panels:
                self._sync_param_widgets()

        def _append_transcript(self, who_tag, who_label, text):
            self.transcript.configure(state="normal")
            if self.transcript.index("end-1c") != "1.0":
                self.transcript.insert("end", "\n")
            self.transcript.insert("end", who_label + "\n", who_tag)
            self.transcript.insert("end", text, "body")
            self.transcript.configure(state="disabled")
            self.transcript.see("end")

        def _append_stream(self, piece):
            self.transcript.configure(state="normal")
            self.transcript.insert("end", piece, "body")
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
            text = self.input_var.get().strip()
            if not model_name:
                self._show_error("Choose a model first (Models tab to download one).")
                return
            if not text:
                return
            self.input_var.set("")
            self.session.set_system(self.system_var.get())
            self.session.params = dict(self.params)
            self.session.add_user(text)
            self._append_transcript("user", "You", text)
            self._append_transcript("bot", "Assistant", "")

            self._generating = True
            self._stop_flag = False
            self.send_btn.state(["disabled"])
            self.stop_btn.state(["!disabled"])
            self._set_status("Generating…", kind="working")
            self.result_lbl.configure(text="")

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
                    self._show_error(err)
                elif self._stop_flag:
                    self._set_status("stopped", kind="idle")
                else:
                    self._show_ok("Done.")

            threading.Thread(target=worker, daemon=True).start()

        def _stop(self):
            self._stop_flag = True

        def _new_chat(self):
            self.session = ChatSession(
                system=self.system_var.get() if hasattr(self, "system_var")
                else prompts_mod.default_prompt(),
                params=dict(self.params), max_context_tokens=4096)
            if hasattr(self, "transcript"):
                self.transcript.configure(state="normal")
                self.transcript.delete("1.0", "end")
                self.transcript.configure(state="disabled")
            self._clear_result()

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
                self._show_ok(f"Saved chat → {path}")
            except LocalLLMError as exc:
                self._show_error(exc)

        def _load_chat(self):
            path = filedialog.askopenfilename(
                title="Load chat (JSON)",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            if not path:
                return
            try:
                self.session = ChatSession.load(path, max_context_tokens=4096)
            except LocalLLMError as exc:
                self._show_error(exc)
                return
            if hasattr(self, "system_var"):
                self.system_var.set(self.session.system)
            if hasattr(self, "transcript"):
                self.transcript.configure(state="normal")
                self.transcript.delete("1.0", "end")
                self.transcript.configure(state="disabled")
                for m in self.session.messages:
                    if m["role"] == "user":
                        self._append_transcript("user", "You", m["content"])
                    else:
                        self._append_transcript("bot", "Assistant", m["content"])
            self._show_ok(f"Loaded chat ← {path}")

        # =================================================================
        # PARAMETERS
        # =================================================================
        def _panel_params(self, parent):
            self._param_vars = {}
            self._param_labels = {}
            grid = ttk.Frame(parent, style="TFrame")
            grid.pack(fill="x", pady=4)
            for i, (key, label, lo, hi, res, is_int) in enumerate(PARAM_SPECS):
                ttk.Label(grid, text=label, width=18, anchor="w").grid(
                    row=i, column=0, sticky="w", pady=4)
                var = tk.DoubleVar(value=float(self.params.get(key, 0)))
                self._param_vars[key] = (var, is_int)
                scale = ttk.Scale(grid, from_=lo, to=hi, variable=var,
                                  orient="horizontal", length=320,
                                  command=lambda v, k=key: self._on_param_change(k))
                scale.grid(row=i, column=1, sticky="we", padx=8)
                vlbl = ttk.Label(grid, text=self._fmt_param(key), width=10,
                                 anchor="e")
                vlbl.grid(row=i, column=2, sticky="e")
                self._param_labels[key] = vlbl
            grid.columnconfigure(1, weight=1)

            btns = ttk.Frame(parent, style="TFrame")
            btns.pack(fill="x", pady=(10, 0))
            ttk.Button(btns, text="Reset to defaults",
                       command=self._reset_params).pack(side="left")
            ttk.Label(parent, style="Muted.TLabel",
                      text="Parameters are saved automatically and applied to "
                           "every generation.").pack(anchor="w", pady=(8, 0))

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
            self._show_ok("Parameters reset to defaults.")

        # =================================================================
        # MODELS
        # =================================================================
        def _panel_models(self, parent):
            dirrow = ttk.Frame(parent, style="TFrame")
            dirrow.pack(fill="x")
            self.modeldir_lbl = ttk.Label(dirrow, style="Sub.TLabel",
                                          text="Models folder: " +
                                          models_mod.models_dir())
            self.modeldir_lbl.pack(side="left")
            ttk.Button(dirrow, text="Open folder",
                       command=lambda: open_in_file_manager(
                           models_mod.models_dir(create=True))).pack(side="right")

            ttk.Label(parent, text="Installed", style="Header.TLabel").pack(
                anchor="w", pady=(10, 2))
            ibody = ttk.Frame(parent, style="TFrame")
            ibody.pack(fill="both", expand=True)
            self.installed_list = tk.Listbox(ibody, height=6, activestyle="none",
                                             exportselection=False)
            isb = ttk.Scrollbar(ibody, orient="vertical",
                                command=self.installed_list.yview)
            self.installed_list.configure(yscrollcommand=isb.set)
            isb.pack(side="right", fill="y")
            self.installed_list.pack(side="left", fill="both", expand=True)
            self.track(self.installed_list, "listbox")
            ttk.Button(parent, text="Refresh",
                       command=self._refresh_installed_list).pack(anchor="w", pady=4)

            ttk.Separator(parent).pack(fill="x", pady=8)
            ttk.Label(parent, text="Download from catalog",
                      style="Header.TLabel").pack(anchor="w")
            crow = ttk.Frame(parent, style="TFrame")
            crow.pack(fill="x", pady=4)
            self.catalog_var = tk.StringVar()
            names = [e.name for e in models_mod.catalog()]
            self.catalog_combo = ttk.Combobox(crow, textvariable=self.catalog_var,
                                              state="readonly", width=44,
                                              values=names)
            if names:
                self.catalog_var.set(names[0])
            self.catalog_combo.pack(side="left", padx=(0, 8))
            self.dl_btn = ttk.Button(crow, text="Download", style="Accent.TButton",
                                     command=self._download_model)
            self.dl_btn.pack(side="left")
            self.dl_progress = ttk.Progressbar(parent, orient="horizontal",
                                               mode="determinate", maximum=100)
            self.dl_progress.pack(fill="x", pady=(8, 2))
            self.catalog_note = ttk.Label(parent, style="Muted.TLabel",
                                          wraplength=760, justify="left", text="")
            self.catalog_note.pack(anchor="w")
            self.catalog_combo.bind("<<ComboboxSelected>>",
                                    lambda e: self._update_catalog_note())
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
            self.dl_progress.configure(value=0)
            self._set_status("Downloading…", kind="working")
            self.result_lbl.configure(text="")

            def progress(done, total):
                pct = int(done * 100 / total) if total else 0
                self.after(0, lambda: self.dl_progress.configure(value=pct))

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
                    self._show_error(err)
                    return
                self.dl_progress.configure(value=100)
                self._show_ok(f"Downloaded → {dest}")
                self._refresh_installed_list()
                self._refresh_models_dropdown()

            threading.Thread(target=worker, daemon=True).start()

        # =================================================================
        # PROMPTS
        # =================================================================
        def _panel_prompts(self, parent):
            body = ttk.Frame(parent, style="TFrame")
            body.pack(fill="both", expand=True)
            left = ttk.Frame(body, style="TFrame")
            left.pack(side="left", fill="y")
            self.prompt_list = tk.Listbox(left, height=12, activestyle="none",
                                          exportselection=False, width=24)
            self.prompt_list.pack(side="left", fill="y")
            self.prompt_list.bind("<<ListboxSelect>>",
                                  lambda e: self._on_prompt_select())
            self.track(self.prompt_list, "listbox")

            right = ttk.Frame(body, style="TFrame")
            right.pack(side="left", fill="both", expand=True, padx=(10, 0))
            ttk.Label(right, text="Name", style="Sub.TLabel").pack(anchor="w")
            self.prompt_name_var = tk.StringVar()
            ttk.Entry(right, textvariable=self.prompt_name_var).pack(fill="x")
            ttk.Label(right, text="System prompt", style="Sub.TLabel").pack(
                anchor="w", pady=(8, 0))
            self.prompt_text = tk.Text(right, wrap="word", height=10)
            self.prompt_text.pack(fill="both", expand=True)
            self.track(self.prompt_text, "text")

            btns = ttk.Frame(right, style="TFrame")
            btns.pack(fill="x", pady=6)
            ttk.Button(btns, text="Save preset", style="Accent.TButton",
                       command=self._save_prompt).pack(side="left")
            ttk.Button(btns, text="Delete", command=self._delete_prompt).pack(
                side="left", padx=6)
            ttk.Button(btns, text="Use in chat",
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
            self.prompt_name_var.set(name)
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", text)

        def _save_prompt(self):
            name = self.prompt_name_var.get().strip()
            text = self.prompt_text.get("1.0", "end").strip()
            try:
                prompts_mod.save_custom(name, text)
            except LocalLLMError as exc:
                self._show_error(exc)
                return
            self._refresh_prompt_list()
            self._show_ok(f"Saved preset '{name}'.")

        def _delete_prompt(self):
            name = self.prompt_name_var.get().strip()
            try:
                prompts_mod.delete_custom(name)
            except LocalLLMError as exc:
                self._show_error(exc)
                return
            self._refresh_prompt_list()
            self._show_ok(f"Deleted preset '{name}'.")

        def _use_prompt(self):
            text = self.prompt_text.get("1.0", "end").strip()
            self.session.set_system(text)
            if hasattr(self, "system_var"):
                self.system_var.set(text)
            self._show_ok("Applied to the current chat's system prompt.")

        # ---- About
        def _about(self):
            win = tk.Toplevel(self)
            win.title("About LocalLLM Studio")
            win.configure(bg=self._pal()["bg"])
            win.resizable(False, False)
            frm = ttk.Frame(win, style="TFrame", padding=18)
            frm.pack(fill="both", expand=True)
            ttk.Label(frm, text="LocalLLM Studio", style="Header.TLabel").pack(anchor="w")
            ttk.Label(frm, text=f"Version {APP_VERSION}", style="Sub.TLabel").pack(
                anchor="w", pady=(0, 8))
            ttk.Label(frm, style="TLabel", justify="left", wraplength=400,
                      text="Run open-weight GGUF language models fully offline on "
                           "your own hardware via llama.cpp.\n\n"
                           "100% AI-built, open source, published on QuickOpen.\n"
                           "Nothing is ever uploaded anywhere.").pack(anchor="w")
            ttk.Label(frm, style="Sub.TLabel", justify="left", wraplength=400,
                      text="Licensed under Apache-2.0. Built on llama-cpp-python "
                           "(MIT). You bring your own model file.").pack(
                anchor="w", pady=(8, 4))
            ttk.Button(frm, text="Close", command=win.destroy).pack(anchor="e")
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

    Returns 0 on a clean exit (including when there is no display, where it
    prints a short message and returns instead of raising).
    """
    try:
        import tkinter as tk  # noqa: F401 - just probing availability
    except Exception as exc:
        sys.stderr.write(f"{APP_NAME}: tkinter is not available ({exc}).\n")
        return 0

    # Headless guard: building a root can fail with no display (TclError).
    try:
        App = build_app()
        app = App()
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

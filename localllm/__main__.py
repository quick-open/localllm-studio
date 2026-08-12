"""Command-line interface: ``python -m localllm <command> ...``.

Commands
  models list                      list installed GGUF models + the models dir
  models catalog                   show the downloadable curated catalog
  models download <name>           download a catalog model (explicit network)
  complete --model M --prompt P    print a completion (``--stream`` to stream)
  chat --model M [--system S]      interactive REPL (/reset /save /system /exit)

Every command exits cleanly (code 2) with a friendly message on
:class:`LocalLLMError` -- in particular when the native library or a model is
missing.  Building the parser never imports the native library.
"""

from __future__ import annotations

import argparse
import sys

from .errors import LocalLLMError
from . import __version__


# ---------------------------------------------------------------------------
# Parser (import-safe: touches no native code)
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="localllm",
        description="Run local GGUF LLMs fully offline (chat + completions).",
    )
    p.add_argument("--version", action="version",
                   version=f"LocalLLM Studio {__version__}")
    sub = p.add_subparsers(dest="command")

    # models ---------------------------------------------------------------
    pm = sub.add_parser("models", help="list / download models")
    msub = pm.add_subparsers(dest="subcommand")
    msub.add_parser("list", help="list installed GGUF models")
    msub.add_parser("catalog", help="list downloadable catalog models")
    pdl = msub.add_parser("download", help="download a catalog model")
    pdl.add_argument("name", help="catalog model name (see `models catalog`)")

    # shared sampling options
    def add_sampling(sp):
        sp.add_argument("--model", "-m", required=True,
                        help="installed model name or a path to a .gguf file")
        sp.add_argument("--temp", "--temperature", type=float, default=None,
                        dest="temperature")
        sp.add_argument("--top-p", type=float, default=None, dest="top_p")
        sp.add_argument("--top-k", type=int, default=None, dest="top_k")
        sp.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
        sp.add_argument("--repeat-penalty", type=float, default=None,
                        dest="repeat_penalty")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--n-ctx", type=int, default=4096, dest="n_ctx")
        sp.add_argument("--n-gpu-layers", type=int, default=0,
                        dest="n_gpu_layers")
        sp.add_argument("--threads", type=int, default=None, dest="threads")

    # complete -------------------------------------------------------------
    pc = sub.add_parser("complete", help="print a single completion")
    add_sampling(pc)
    pc.add_argument("--prompt", "-p", required=True, help="prompt text")
    pc.add_argument("--stream", action="store_true", help="stream tokens")

    # chat -----------------------------------------------------------------
    pch = sub.add_parser("chat", help="interactive chat REPL")
    add_sampling(pch)
    pch.add_argument("--system", "-s", default=None, help="system prompt text")
    pch.add_argument("--preset", default=None,
                     help="preset: assistant | coding | creative")
    pch.add_argument("--no-stream", action="store_true",
                     help="disable token streaming")
    return p


def _params_from_args(a):
    keys = ("temperature", "top_p", "top_k", "max_tokens", "repeat_penalty", "seed")
    return {k: getattr(a, k) for k in keys if getattr(a, k, None) is not None}


def _new_engine(a):
    from .engine import LlamaEngine
    return LlamaEngine(
        a.model, n_ctx=a.n_ctx, n_threads=a.threads,
        n_gpu_layers=a.n_gpu_layers)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
def cmd_models_list(a):
    from . import models as M
    d = M.models_dir()
    installed = M.list_models(d)
    print(f"Models directory: {d}")
    if not installed:
        print("  (no .gguf models installed yet)")
        print("  Add a .gguf file there, or run `localllm models download <name>`.")
        return 0
    for mi in installed:
        print(f"  {mi.name}  ({M.human_size(mi.size)})")
    return 0


def cmd_models_catalog(a):
    from . import models as M
    print("Downloadable catalog (permissively licensed GGUF models):")
    for e in M.catalog():
        print(f"  {e.name}")
        print(f"      {e.params} params, {e.quant}, ~{M.human_size(e.approx_size)}, "
              f"{e.license}")
        if e.note:
            print(f"      {e.note}")
    print("\nDownload with:  localllm models download <name>")
    return 0


def cmd_models_download(a):
    from . import models as M

    last = [-1]

    def progress(done, total):
        if total:
            pct = int(done * 100 / total)
            if pct != last[0]:
                last[0] = pct
                sys.stderr.write(
                    f"\rDownloading {a.name}: {pct}%  "
                    f"({M.human_size(done)}/{M.human_size(total)})")
                sys.stderr.flush()
        else:
            sys.stderr.write(f"\rDownloading {a.name}: {M.human_size(done)}")
            sys.stderr.flush()

    dest = M.download(a.name, progress=progress)
    sys.stderr.write("\n")
    print(f"Saved {dest}")
    return 0


def cmd_complete(a):
    eng = _new_engine(a)
    params = _params_from_args(a)
    if a.stream:
        for piece in eng.stream_complete(a.prompt, **params):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        print(eng.complete(a.prompt, **params))
    return 0


def cmd_chat(a):
    from .chatsession import ChatSession
    from . import prompts

    system = a.system
    params = _params_from_args(a)
    sess = ChatSession(system=system or "", params=params, model=a.model)
    if a.preset:
        sess.apply_preset(a.preset, prompts_module=prompts)
        params = dict(sess.params)
    if not sess.system:
        sess.set_system(prompts.default_prompt())

    eng = _new_engine(a)
    stream = not a.no_stream

    print("LocalLLM Studio chat. Commands: /reset  /system <text>  "
          "/save <file>  /exit")
    print(f"Model: {a.model}\n")
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.rstrip("\n")
        if not line.strip():
            continue
        if line.strip() in ("/exit", "/quit"):
            break
        if line.strip() == "/reset":
            sess.reset()
            print("(history cleared)")
            continue
        if line.startswith("/system"):
            sess.set_system(line[len("/system"):].strip())
            print("(system prompt updated)")
            continue
        if line.startswith("/save"):
            path = line[len("/save"):].strip()
            if not path:
                print("(usage: /save <file.json|file.md>)")
                continue
            try:
                sess.save(path)
                print(f"(saved {path})")
            except LocalLLMError as exc:
                print(f"(error: {exc})")
            continue

        sess.add_user(line)
        msgs = sess.build_messages()
        sys.stdout.write("bot> ")
        sys.stdout.flush()
        reply_parts = []
        if stream:
            for piece in eng.stream_chat(msgs, **sess.params):
                reply_parts.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            reply = eng.chat(msgs, **sess.params)
            reply_parts.append(reply)
            print(reply)
        sess.add_assistant("".join(reply_parts))
    return 0


def _dispatch(a):
    if a.command == "models":
        sub = getattr(a, "subcommand", None)
        if sub == "list":
            return cmd_models_list(a)
        if sub == "catalog":
            return cmd_models_catalog(a)
        if sub == "download":
            return cmd_models_download(a)
        # bare `models` -> show installed
        return cmd_models_list(a)
    if a.command == "complete":
        return cmd_complete(a)
    if a.command == "chat":
        return cmd_chat(a)
    return None


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    try:
        rc = _dispatch(args)
    except LocalLLMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130
    return 0 if rc is None else rc


if __name__ == "__main__":
    sys.exit(main())

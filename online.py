import argparse
import sys
import time
from datetime import datetime
from ddgs import DDGS
from llama_cpp import Llama

# ── Optional rich terminal UI ──────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.spinner import Spinner
    from rich.live import Live
    from rich.rule import Rule
    from rich.theme import Theme
    from rich import print as rprint
    RICH = True
except ImportError:
    RICH = False

THEME = Theme({
    "info":    "bold cyan",
    "success": "bold green",
    "warn":    "bold yellow",
    "error":   "bold red",
    "dim":     "dim white",
    "ai":      "bold bright_white",
})
console = Console(theme=THEME) if RICH else None


# ── Helpers ────────────────────────────────────────────────────────────────────

def cprint(msg, style="", markup=True):
    if RICH:
        console.print(msg, style=style, markup=markup)
    else:
        print(msg)


def section(title):
    if RICH:
        console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="dim"))
    else:
        print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ── Web Search ─────────────────────────────────────────────────────────────────

def search_web(query: str, max_results: int = 5, region: str = "wt-wt") -> list[dict]:
    """
    Returns a list of {title, url, body} dicts from DuckDuckGo.
    Never raises — returns empty list on failure.
    """
    cprint(f"\n[info]🔍 Searching:[/info] [dim]{query}[/dim]") if RICH else print(f"[*] Searching: {query}")
    results = []
    try:
        with DDGS() as ddg:
            raw = ddg.text(query=query, max_results=max_results, region=region)
            for r in raw:
                results.append({
                    "title": r.get("title", "").strip(),
                    "url":   r.get("href", ""),
                    "body":  r.get("body",  "").strip(),
                })
    except Exception as e:
        cprint(f"[warn]⚠ Search error:[/warn] {e}") if RICH else print(f"[!] Search error: {e}")
    return results


def build_context_block(results: list[dict]) -> str:
    if not results:
        return "No web search results available."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['body']}")
    return "\n\n".join(lines)


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(model_path: str, n_ctx: int, n_gpu_layers: int, n_threads: int) -> Llama:
    cprint("\n[info]⚙ Loading model…[/info]") if RICH else print("[*] Loading model…")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,   # 0 = CPU-only, -1 = all layers on GPU
            n_threads=n_threads,
            verbose=False,
        )
    except Exception as e:
        cprint(f"[error]✖ Failed to load model:[/error] {e}") if RICH else print(f"[!] Failed: {e}")
        sys.exit(1)
    cprint("[success]✔ Model ready.[/success]\n") if RICH else print("[+] Model ready.\n")
    return llm


# ── Streaming inference ────────────────────────────────────────────────────────

def stream_response(llm: Llama, messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, dict]:
    """
    Streams tokens to stdout and returns (full_text, usage_dict).
    """
    full_text = ""
    usage = {}

    if RICH:
        console.print("\n[bold bright_white]Assistant ▸[/bold bright_white] ", end="")

    stream = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content", "")
        if token:
            full_text += token
            if RICH:
                console.print(token, end="", markup=False)
            else:
                print(token, end="", flush=True)
        # capture usage from final chunk
        if chunk["choices"][0].get("finish_reason"):
            usage = chunk.get("usage", {})

    print()  # newline after stream ends
    return full_text, usage


# ── Token / timing stats ───────────────────────────────────────────────────────

def print_stats(usage: dict, elapsed: float):
    if not usage:
        return
    p_tok = usage.get("prompt_tokens", "?")
    c_tok = usage.get("completion_tokens", "?")
    t_tok = usage.get("total_tokens", "?")
    tps   = round(usage.get("completion_tokens", 0) / elapsed, 1) if elapsed > 0 else "?"
    if RICH:
        console.print(
            f"\n[dim]  ⏱ {elapsed:.2f}s · "
            f"prompt {p_tok} tok · "
            f"completion {c_tok} tok · "
            f"total {t_tok} tok · "
            f"{tps} tok/s[/dim]"
        )
    else:
        print(f"\n[stats] {elapsed:.2f}s | prompt={p_tok} | completion={c_tok} | {tps} tok/s")


# ── Conversation session ───────────────────────────────────────────────────────

class Session:
    def __init__(self, llm: Llama, args):
        self.llm   = llm
        self.args  = args
        self.history: list[dict] = []   # pure chat history (no system msg)
        self.turn  = 0

    def _build_messages(self, system_content: str) -> list[dict]:
        return [{"role": "system", "content": system_content}] + self.history

    def ask(self, user_input: str):
        self.turn += 1

        # ── web search ──────────────────────────────────────────────────────
        search_needed = not self.args.no_search
        if search_needed:
            results = search_web(user_input, max_results=self.args.results, region=self.args.region)
        else:
            results = []

        context_block = build_context_block(results)

        # ── print sources ───────────────────────────────────────────────────
        if results:
            if RICH:
                console.print("\n[dim]Sources found:[/dim]")
                for r in results:
                    console.print(f"  [dim cyan]• {r['title']}[/dim cyan]  [dim]{r['url']}[/dim]")
            else:
                print("\nSources:")
                for r in results:
                    print(f"  • {r['title']}  {r['url']}")

        # ── system prompt ────────────────────────────────────────────────────
        date_str = datetime.now().strftime("%A, %B %d, %Y %H:%M")
        system_msg = (
            f"You are a knowledgeable, concise AI assistant. "
            f"Today is {date_str}.\n\n"
            f"Web Search Context (cite [1], [2] … when using):\n"
            f"{context_block}\n\n"
            f"Instructions:\n"
            f"- Use the web context above when it is relevant.\n"
            f"- If the context is insufficient, rely on your own knowledge and say so.\n"
            f"- Keep answers clear and well-structured.\n"
            f"- Do NOT repeat the search results verbatim; synthesise them."
        )

        # ── append user turn to history ──────────────────────────────────────
        self.history.append({"role": "user", "content": user_input})
        messages = self._build_messages(system_msg)

        # ── generate ─────────────────────────────────────────────────────────
        section(f"Turn {self.turn}")
        t0 = time.perf_counter()
        reply, usage = stream_response(
            self.llm, messages,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
        )
        elapsed = time.perf_counter() - t0
        print_stats(usage, elapsed)

        # ── append assistant turn to history ─────────────────────────────────
        self.history.append({"role": "assistant", "content": reply})

        # ── trim history to avoid context overflow ────────────────────────────
        max_pairs = self.args.history_turns
        if len(self.history) > max_pairs * 2:
            self.history = self.history[-(max_pairs * 2):]


# ── CLI entry-point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="🔍  Local LLM + Live Web Search — enhanced CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("-m", "--model",       required=True,  help="Path to .gguf model file")
    p.add_argument("--n-ctx",            type=int, default=4096,  help="Context window size (tokens)")
    p.add_argument("--n-gpu-layers",     type=int, default=0,     help="GPU layers (-1 = all)")
    p.add_argument("--n-threads",        type=int, default=4,     help="CPU threads for inference")
    # Generation
    p.add_argument("--max-tokens",       type=int,   default=512,  help="Max tokens per reply")
    p.add_argument("--temperature",      type=float, default=0.2,  help="Sampling temperature")
    # Search
    p.add_argument("--results",          type=int,  default=5,    help="Web results to fetch per query")
    p.add_argument("--region",           default="wt-wt",          help="DDGS region code (e.g. us-en)")
    p.add_argument("--no-search",        action="store_true",      help="Disable web search entirely")
    # Session
    p.add_argument("--history-turns",    type=int, default=6,     help="Conversation turns to keep in context")
    p.add_argument("-p", "--prompt",     default=None,             help="Single-shot prompt (skips REPL loop)")
    return p.parse_args()


def repl(session: Session):
    if RICH:
        console.print(Panel(
            "[bold]Local LLM + Live Web Search[/bold]\n"
            "[dim]Type your question and press Enter. "
            "Commands: [bold]/clear[/bold] · [bold]/history[/bold] · [bold]/nosearch[/bold] · [bold]/quit[/bold][/dim]",
            style="cyan",
            expand=False,
        ))
    else:
        print("\n=== Local LLM + Live Web Search ===")
        print("Commands: /clear · /history · /nosearch · /quit\n")

    while True:
        try:
            if RICH:
                user_input = console.input("\n[bold cyan]You ▸[/bold cyan] ").strip()
            else:
                user_input = input("\nYou ▸ ").strip()
        except (KeyboardInterrupt, EOFError):
            cprint("\n\n[dim]Bye![/dim]") if RICH else print("\nBye!")
            break

        if not user_input:
            continue

        # ── in-REPL commands ─────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "/q"):
            cprint("\n[dim]Exiting. Goodbye![/dim]") if RICH else print("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            session.history.clear()
            session.turn = 0
            cprint("[success]✔ History cleared.[/success]") if RICH else print("[+] History cleared.")
            continue
        elif user_input.lower() == "/history":
            if not session.history:
                cprint("[dim]No history yet.[/dim]") if RICH else print("No history yet.")
            for msg in session.history:
                role = msg["role"].upper()
                cprint(f"[dim]{role}:[/dim] {msg['content'][:120]}…") if RICH else print(f"{role}: {msg['content'][:120]}")
            continue
        elif user_input.lower() == "/nosearch":
            session.args.no_search = not session.args.no_search
            state = "OFF" if session.args.no_search else "ON"
            cprint(f"[warn]Web search toggled {state}.[/warn]") if RICH else print(f"[!] Web search {state}")
            continue

        session.ask(user_input)


def main():
    args = parse_args()
    llm  = load_model(args.model, args.n_ctx, args.n_gpu_layers, args.n_threads)
    session = Session(llm, args)

    if args.prompt:
        # single-shot mode (original behaviour, now with streaming + stats)
        session.ask(args.prompt)
    else:
        # interactive REPL
        repl(session)


if __name__ == "__main__":
    main()

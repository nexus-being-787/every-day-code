import time
from datetime import datetime
import streamlit as st

# Handle duckduckgo import compatibility
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS

from llama_cpp import Llama

# ── Streamlit Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Local AI + Web Search",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Styling (ChatGPT Dark Vibe) ───────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0e0e11;
    }
    .source-box {
        background-color: #1e1e24;
        border-left: 3px solid #00adb5;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .stats-text {
        color: #888888;
        font-size: 0.8rem;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Web Search Helper ──────────────────────────────────────────────────────────
def search_web(query: str, max_results: int = 5, region: str = "wt-wt") -> list[dict]:
    results = []
    try:
        with DDGS() as ddg:
            raw = ddg.text(query=query, max_results=max_results, region=region)
            for r in raw:
                results.append({
                    "title": r.get("title", "").strip(),
                    "url": r.get("href", ""),
                    "body": r.get("body", "").strip(),
                })
    except Exception as e:
        st.warning(f"Search warning: {e}")
    return results


def build_context_block(results: list[dict]) -> str:
    if not results:
        return "No web search results available."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['body']}")
    return "\n\n".join(lines)


# ── Model Caching (Loads Model Once into Memory) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(model_path: str, n_ctx: int, n_gpu_layers: int, n_threads: int):
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=False,
    )


# ── Sidebar Configuration ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Model Settings")

    model_path = st.text_input(
        "GGUF Model Path",
        value="/home/titan/Downloads/LFM2.5-VL-450M-Q4_K_M.gguf"
    )

    enable_search = st.toggle("Enable Web Search", value=True)
    num_results = st.slider("Search Results", 1, 10, 5)

    st.divider()

    n_ctx = st.select_slider("Context Size (tokens)", options=[2048, 4096, 8192, 16384], value=4096)
    n_gpu_layers = st.number_input("GPU Layers (-1 = All, 0 = CPU)", value=0)
    n_threads = st.number_input("CPU Threads", value=4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max Tokens per Reply", 128, 4096, 1024)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Load Model ─────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading Llama-cpp model into memory..."):
        llm = get_model(model_path, n_ctx, n_gpu_layers, n_threads)
except Exception as e:
    st.error(f"Failed to load GGUF model from path: `{model_path}`\n\n**Error:** {e}")
    st.stop()


# ── Initialize State ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Local Assistant")
st.caption("Powered by `llama-cpp-python` & DuckDuckGo Search")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"[{src['title']}]({src['url']})")
        if "stats" in msg:
            st.markdown(f"<div class='stats-text'>{msg['stats']}</div>", unsafe_allow_html=True)


# ── Chat Input & Streaming ────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question..."):
    # Render user message instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Perform web search if toggled
    sources = []
    if enable_search:
        with st.status("🔍 Searching the web...", expanded=False) as status:
            sources = search_web(prompt, max_results=num_results)
            if sources:
                status.update(label=f"Found {len(sources)} sources!", state="complete")
            else:
                status.update(label="No relevant web results found.", state="complete")

    # Build system prompt with web context
    context_block = build_context_block(sources)
    date_str = datetime.now().strftime("%A, %B %d, %Y %H:%M")
    system_msg = (
        f"You are a knowledgeable, concise AI assistant. Today is {date_str}.\n\n"
        f"Web Search Context (cite [1], [2] … when using):\n{context_block}\n\n"
        f"Instructions:\n"
        f"- Use the web context above when it is relevant.\n"
        f"- If the context is insufficient, rely on your own knowledge and say so.\n"
        f"- Keep answers clear and well-structured.\n"
        f"- Do NOT repeat the search results verbatim; synthesise them."
    )

    # Assemble conversation payload
    api_messages = [{"role": "system", "content": system_msg}]
    for m in st.session_state.messages[-12:]:  # limit history depth
        api_messages.append({"role": m["role"], "content": m["content"]})

    # Render Assistant streaming block
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        t0 = time.perf_counter()

        # Stream response chunks
        stream = llm.create_chat_completion(
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                full_response += token
                message_placeholder.markdown(full_response + "▌")

        elapsed = time.perf_counter() - t0
        message_placeholder.markdown(full_response)

        # Show web sources if available
        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"• [{s['title']}]({s['url']})")

        # Stats display
        tps = round(len(full_response.split()) * 1.3 / elapsed, 1) if elapsed > 0 else 0
        stats_info = f"⏱ {elapsed:.2f}s | ~{tps} tok/s"
        st.markdown(f"<div class='stats-text'>{stats_info}</div>", unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
        "stats": stats_info
    })

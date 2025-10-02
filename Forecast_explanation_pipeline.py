# fast_testing.py
import os, contextlib, time
from llama_cpp import Llama
from datetime import date

# -------- settings you can tweak --------
MODEL_PATH = "/Users/jubinmohanty/Desktop/rag_pipeline/phi1_5.gguf"
N_CTX = 1024                 # model supports up to 2048; 1024 is a good start
RESERVE_TOKENS = 32          # safety buffer so we don't hit the ceiling
WARMUP = True                # tiny warmup for stable first-run latency
# ---------------------------------------

# keep the backend quiet
os.environ.setdefault("GGML_LOG_LEVEL", "ERROR")
os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")
os.environ.setdefault("LLAMA_DISABLE_PERF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

@contextlib.contextmanager
def silence_backend_stdio():
    devnull = open(os.devnull, "w")
    old_out, old_err = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(old_out, 1); os.dup2(old_err, 2)
        os.close(old_out); os.close(old_err); devnull.close()

def build_llm():
    kwargs = dict(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_batch=1024,      # speeds prompt eval on Metal/CPU
        n_ubatch=256,      # reduce per-token latency
        n_threads=6,       # try 4–8 on M2; keep what’s fastest
        use_mmap=True,
        logits_all=False,
        verbose=False,
        seed=42,
        cache_prompt=True, # important for fast continuations
    )
    try:
        return Llama(gpu_layers=999, **kwargs)   # newer wheels
    except TypeError:
        return Llama(n_gpu_layers=999, **kwargs) # older wheels

with silence_backend_stdio():
    llm = build_llm()
    if WARMUP:
        _ = llm.create_completion(prompt="Hi", max_tokens=1, temperature=0.0)

def tokens_available_for_generation(prompt_text: str) -> int:
    """How many tokens we can still generate without overflowing context."""
    ptoks = len(llm.tokenize(prompt_text.encode("utf-8")))
    return max(0, N_CTX - ptoks - RESERVE_TOKENS)

def generate_full(prompt: str,
                  chunk_tokens: int = 256,
                  max_total_tokens: int = 300,
                  temperature: float = 0.25,
                  top_k: int = 40,
                  top_p: float = 0.95,
                  repeat_penalty: float = 1.05,
                  stop=None):
    """
    Keep calling create_completion in chunks until the model stops naturally
    or we hit max_total_tokens. Returns (full_text, gen_seconds, n_tokens).
    """
    if stop is None:
        stop = []  # no custom stop strings; let EOS end it

    full_text_parts = []
    total_gen_tokens = 0
    t0 = time.perf_counter()

    cur_prompt = prompt
    while total_gen_tokens < max_total_tokens:
        avail = tokens_available_for_generation(cur_prompt)
        if avail <= 0:
            print(f"[debug] No room left in context: prompt_tokens="
                  f"{len(llm.tokenize(cur_prompt.encode('utf-8')))}, N_CTX={N_CTX}")
            break

        to_request = min(chunk_tokens, avail, max_total_tokens - total_gen_tokens)

        with silence_backend_stdio():
            out = llm.create_completion(
                prompt=cur_prompt,
                max_tokens=to_request,
                temperature=temperature,
                top_k=top_k, top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=stop,
            )

        choices = out.get("choices") or []
        if not choices or "text" not in choices[0]:
            print("[debug] No choices/text. Raw:", out)
            break

        piece = choices[0].get("text") or ""
        finish = choices[0].get("finish_reason")
        usage = out.get("usage") or {}
        n_comp = usage.get("completion_tokens") or 0

        # Debug: show why this chunk stopped and prompt size now
        ptoks = len(llm.tokenize(cur_prompt.encode("utf-8")))
        print(f"[debug] finish_reason={finish!r}, this_chunk={n_comp} tokens, "
              f"prompt_tokens={ptoks}, avail_before={avail}, requested={to_request}")

        full_text_parts.append(piece)
        total_gen_tokens += n_comp

        # continue only if we hit the per-call length cap
        if finish != "length":
            break

        # continue from where we left off
        cur_prompt = prompt + "".join(full_text_parts)

    secs = time.perf_counter() - t0
    return "".join(full_text_parts).strip(), secs, total_gen_tokens

# ---------- Your forecast-explanation prompt (optimized & compact) ----------
PROMPT_TEMPLATE = """You are a statistical analyst and data scientist. Provide a concise explanation (3–6 bullet points plus a short paragraph) for the forecasted sales.

Context:
- SKU: {sku}
- Country: {country}
- Forecast date: {f_date}
- Forecasted sales: {f_sales:.2f}

Historical (computed):
- Observations: {count} weeks
- Mean weekly sales: {mean:.2f}
- Std dev: {std:.2f}
- Latest observed week: {latest:.2f}
- Trend (slope/wk): {slope_per_week:.4f}
- Holiday lift (vs non-holiday): {holiday_lift:.1%}
- Promo lift (vs non-promo): {promo_lift:.1%}

Related notes (optional):
{retrieved_block}

Requirements:
1) Quantify why the forecast is ↑/↓/≈ vs recent weeks using % difference from latest and/or mean.
2) Identify top 2 drivers (trend / holiday / promotions / recent spike/drop).
3) Give confidence (LOW / MEDIUM / HIGH) + one-line reason (variance, strong drivers, data sparsity).
4) Suggest 1–2 concrete next steps.
5) ≤ 200 words. Plain language. Start with bullets, end with a brief paragraph.

Now produce the explanation.
"""

def render_prompt(sku, country, forecast_row, stats, retrieved_block="(none)"):
    return PROMPT_TEMPLATE.format(
        sku=sku,
        country=country,
        f_date=forecast_row["date"],
        f_sales=forecast_row["sales"],
        count=stats["count"],
        mean=stats["mean"],
        std=stats["std"],
        latest=stats["latest"],
        slope_per_week=stats["slope_per_week"],
        holiday_lift=stats["holiday_lift"],
        promo_lift=stats["promo_lift"],
        retrieved_block=retrieved_block.strip() if retrieved_block else "(none)",
    )

if __name__ == "__main__":
    # ---- Dummy values you can replace with real ones later ----
    sku = "SKU-12345"
    country = "DE"
    forecast_row = {"date": str(date(2025, 2, 2)), "sales": 295.0}
    stats = {
        "count": 52,
        "mean": 270.4,
        "std": 32.8,
        "latest": 282.0,
        "slope_per_week": 0.9,
        "holiday_lift": 0.12,   # 12%
        "promo_lift": 0.28,     # 28%
    }
    retrieved_block = (
        "- Next promo planned mid-Feb.\n"
        "- Holiday weekend at end-Jan historically adds ~10–15%.\n"
        "- Stock constraints last month now resolved."
    )

    prompt = render_prompt(sku, country, forecast_row, stats, retrieved_block)

    text, secs, n_tokens = generate_full(
        prompt,
        chunk_tokens=256,        # small chunks; we want concise output
        max_total_tokens=300,    # hard cap to keep ≤ ~200 words
        temperature=0.2,         # lower = more focused/concise
        top_k=40,
        top_p=0.9,
        repeat_penalty=1.05,
        stop=["<|endoftext|>"],  # EOS
    )

    print("\n" + "="*80 + "\n")
    print(text)
    if secs > 0:
        rate = (n_tokens / secs) if n_tokens else 0.0
        print(f"\n[gen: {secs:.2f} s | tokens: {n_tokens} | {rate:.2f} tok/s]")
    else:
        print("\n[gen: 0.00 s]")

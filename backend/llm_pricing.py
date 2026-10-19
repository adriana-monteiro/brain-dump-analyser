"""LLM price table and cost arithmetic. Shared by the app (usage recording) and the evals."""

# USD per 1M tokens: (input, cached input, output). Checked against OpenAI's published list prices in Oct 2026;
# prices change, so edit here. A snapshot name such as gpt-4.1-mini-2025-04-14 matches its longest prefix.
# cached=None bills cached tokens at the input price (an overestimate).
PRICES = {
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, None, 8.00),
}
OVERRIDE = None        # (in, cached, out) applied to every model; set by the eval CLI (--price-in / --price-out)


def price_for(model):
    if OVERRIDE: return OVERRIDE
    hits = [k for k in PRICES if (model or "").startswith(k)]
    return PRICES[max(hits, key=len)] if hits else None


def cost_parts(model, prompt_tokens, completion_tokens, cached_tokens=0):
    """(input_usd, cached_input_usd, output_usd) for one call, or None if the model has no price.
    prompt_tokens includes the cached ones, as in the OpenAI usage object."""
    p = price_for(model)
    if p is None: return None
    pin, pcached, pout = p
    cached = min(cached_tokens or 0, prompt_tokens)
    return ((prompt_tokens - cached) * pin / 1e6,
            cached * (pin if pcached is None else pcached) / 1e6,
            completion_tokens * pout / 1e6)
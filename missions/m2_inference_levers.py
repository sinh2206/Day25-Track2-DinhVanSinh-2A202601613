"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
    }


# ── [EXTENSION 4] Reasoning Budget ──────────────────────────────────────────

def reasoning_budget_analysis(verbose: bool = True) -> dict:
    """[EXTENSION 4] Separate cost & energy for is_reasoning=1 vs is_reasoning=0.

    Reasoning queries (~chain-of-thought) consume ~80x more energy than standard
    queries. This function:
      1. Splits traffic into reasoning vs normal.
      2. Calculates $ cost and Wh energy for each group.
      3. Proposes a routing cap: if capping reasoning to 10% of traffic,
         how much $ and Wh is saved?
    """
    from finops.sustainability import wh_per_query

    rows = load_csv("token_usage.csv")

    # Accumulators
    r_cost = r_tokens = r_wh = r_count = 0.0
    n_cost = n_tokens = n_wh = n_count = 0.0

    for r in rows:
        inp = int(num(r["input_tokens"]))
        out = int(num(r["output_tokens"]))
        is_reasoning = bool(int(num(r.get("is_reasoning", "0"))))
        tier = r["route_tier"]
        pin, pout = MODEL_PRICES[tier]
        cost = pricing.request_cost(inp, out, pin, pout)
        tokens = inp + out
        wh = wh_per_query(tokens, is_reasoning=is_reasoning)

        if is_reasoning:
            r_cost += cost; r_tokens += tokens; r_wh += wh; r_count += 1
        else:
            n_cost += cost; n_tokens += tokens; n_wh += wh; n_count += 1

    total_count = r_count + n_count
    total_cost = r_cost + n_cost
    total_wh = r_wh + n_wh

    reasoning_traffic_pct = (r_count / total_count * 100) if total_count else 0
    reasoning_cost_pct = (r_cost / total_cost * 100) if total_cost else 0
    reasoning_wh_pct = (r_wh / total_wh * 100) if total_wh else 0

    # Projection: cap reasoning at 10% of traffic instead of current rate
    cap_pct = 0.10
    current_reasoning_frac = r_count / total_count if total_count else 0
    if current_reasoning_frac > cap_pct:
        reduction_frac = 1.0 - (cap_pct / current_reasoning_frac)
        saved_cost = r_cost * reduction_frac
        saved_wh = r_wh * reduction_frac
    else:
        saved_cost = saved_wh = 0.0

    if verbose:
        print("\n== [Extension 4] Reasoning Budget Analysis ==")
        print(f"{'Metric':<30} {'Reasoning':>12} {'Normal':>12} {'Reasoning%':>12}")
        print("-" * 68)
        print(f"{'Requests':<30} {r_count:>12.0f} {n_count:>12.0f} {reasoning_traffic_pct:>11.1f}%")
        print(f"{'Cost ($/day)':<30} {r_cost:>12.4f} {n_cost:>12.4f} {reasoning_cost_pct:>11.1f}%")
        print(f"{'Energy (Wh/day)':<30} {r_wh:>12.2f} {n_wh:>12.2f} {reasoning_wh_pct:>11.1f}%")
        print(f"{'Avg tokens/request':<30} {(r_tokens/r_count if r_count else 0):>12.0f} {(n_tokens/n_count if n_count else 0):>12.0f}")
        print()
        print(f"  Reasoning uses ~80x more energy per token than normal queries.")
        print(f"  Though reasoning is {reasoning_traffic_pct:.1f}% of traffic,")
        print(f"     it consumes {reasoning_cost_pct:.1f}% of cost and {reasoning_wh_pct:.1f}% of energy.")
        print()
        print(f"  Routing Cap Proposal: limit reasoning to {cap_pct:.0%} of traffic")
        if saved_cost > 0:
            print(f"     -> Potential savings: ${saved_cost:.4f}/day  |  {saved_wh:.2f} Wh/day")
            print(f"     -> Rule: only invoke reasoning when task complexity score > threshold")
        else:
            print(f"     -> Reasoning already at or below {cap_pct:.0%} -- no action needed.")

    return {
        "reasoning_requests": int(r_count),
        "normal_requests": int(n_count),
        "reasoning_cost_daily": round(r_cost, 4),
        "normal_cost_daily": round(n_cost, 4),
        "reasoning_wh_daily": round(r_wh, 2),
        "normal_wh_daily": round(n_wh, 2),
        "reasoning_traffic_pct": round(reasoning_traffic_pct, 1),
        "reasoning_cost_pct": round(reasoning_cost_pct, 1),
        "reasoning_wh_pct": round(reasoning_wh_pct, 1),
        "cap_10pct_saved_cost": round(saved_cost, 4),
        "cap_10pct_saved_wh": round(saved_wh, 2),
    }


if __name__ == "__main__":
    run()
    reasoning_budget_analysis()

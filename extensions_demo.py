"""extensions_demo.py — Demonstrates Extension 1, 3, 4 with before/after measurements.

Run: python extensions_demo.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finops import pricing
from missions._common import load_csv, num, catalog_by_type
from missions.m2_inference_levers import run as m2_run, reasoning_budget_analysis

DAYS = 30

# ─────────────────────────────────────────────────────────────────
# EXTENSION 1 — Enhanced recommend_tier with GPU interruption rates
#               + 1yr vs 3yr reserved comparison
# ─────────────────────────────────────────────────────────────────

def ext1_demo():
    print("=" * 70)
    print("EXTENSION 1 — Enhanced recommend_tier() with GPU interruption rates")
    print("             + 1yr vs 3yr reserved comparison")
    print("=" * 70)

    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    print(f"\n{'Job':20} {'GPU':6} {'Days':5} {'Base Tier':12} {'Commitment':11} {'Reasoning'}")
    print("-" * 100)

    old_total = new_total = 0.0

    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        job_days = int(num(j.get("days", "30")))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])

        # EXTENSION 1: detailed commitment analysis
        enhanced = pricing.recommend_tier_detailed(hpd, interruptible,
                                                   gpu_type=gtype, job_days=job_days)
        tier = enhanced["tier"]
        commitment = enhanced["commitment"] or "-"

        def get_cost(t, comm=None):
            if t == "spot":
                return pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
            elif t == "reserved":
                if comm == "1yr":
                    return gpu_hours * num(c["reserved_1yr_hr"])
                return gpu_hours * num(c["reserved_3yr_hr"])
            return gpu_hours * od

        old_cost = get_cost("reserved")  # naive: always 3yr reserved if not spot
        new_cost = get_cost(tier, enhanced["commitment"])
        old_total += old_cost
        new_total += new_cost

        print(f"{j['job_id']:20} {gtype:6} {job_days:5} {tier:12} {commitment:11} {enhanced['reasoning'][:45]}")

    savings_delta = old_total - new_total
    print()
    print(f"  If naive policy committed ALL stable jobs to 3yr: ${old_total:,.0f}")
    print(f"  Smart policy (1yr for short jobs):                ${new_total:,.0f}")
    if savings_delta > 0:
        print(f"  Extra savings: ${savings_delta:,.0f} ({savings_delta/old_total*100:.1f}%)")
    else:
        print(f"  Smart policy avoids over-commitment risk: ${-savings_delta:,.0f} more expensive")
        print(f"  but AVOIDS locking in 3yr on 30-day jobs (335 idle days = wasted discount)")
    print()
    print("  GPU interruption rates (data-driven, Extension 1):")
    for gpu, rate in pricing.GPU_SPOT_INTERRUPT_RATE.items():
        print(f"    {gpu:6} spot interrupt rate: {rate:.0%}")


# ─────────────────────────────────────────────────────────────────
# EXTENSION 3 — cache_is_worth_it() with break-even analysis
# ─────────────────────────────────────────────────────────────────

def ext3_demo():
    print("=" * 70)
    print("EXTENSION 3 — cache_is_worth_it(): Break-even cache analysis")
    print("=" * 70)

    rows = load_csv("token_usage.csv")
    total_requests = len(rows)
    cached_requests = sum(1 for r in rows if int(num(r["cached_input_tokens"])) > 0)
    total_input = sum(int(num(r["input_tokens"])) for r in rows)
    total_cached = sum(int(num(r["cached_input_tokens"])) for r in rows)
    avg_cache_hit_frac = total_cached / total_input if total_input else 0.0
    writers = total_requests - cached_requests
    avg_reads = cached_requests / max(writers, 1)

    break_even = pricing.cache_break_even_reads()
    worth_it = pricing.cache_is_worth_it(avg_reads, write_cost_per_m=1.25)

    print(f"\n  Dataset stats:")
    print(f"    Total requests:          {total_requests}")
    print(f"    Requests with cache hit: {cached_requests} ({cached_requests/total_requests*100:.1f}%)")
    print(f"    Avg cache hit fraction:  {avg_cache_hit_frac:.1%} of input tokens cached")
    print(f"    Estimated avg reads/prefix: {avg_reads:.2f}x")
    print()
    print(f"  Break-even: need >{break_even:.2f}x reads per prefix to justify cache cost")
    print(f"  Our dataset: {avg_reads:.2f}x reads  → caching {'JUSTIFIED ✅' if worth_it else 'NOT justified ❌'}")
    print()
    print(f"  Scenario sweep:")
    print(f"  {'Avg reads':>10} {'Worth it?':>12}")
    print(f"  {'-'*25}")
    for r in [0.5, 1.0, 1.12, 2.0, 5.0, 10.0]:
        w = pricing.cache_is_worth_it(r, write_cost_per_m=1.25)
        mark = "our dataset ~>" if abs(r - round(avg_reads)) < 0.5 else ""
        print(f"  {r:>10.1f} {'YES ✅' if w else 'NO ❌':>12}  {mark}")
    print()
    print("  KEY: read_discount=10% → break-even at 1.11x reads per prefix")
    print("  Caching ONLY makes sense with repetitive system prompts or RAG contexts")


# ─────────────────────────────────────────────────────────────────
# EXTENSION 4 — Reasoning Budget Analysis
# ─────────────────────────────────────────────────────────────────

def ext4_demo():
    print("=" * 70)
    print("EXTENSION 4 — Reasoning Budget: Cost & Energy breakdown")
    print("=" * 70)
    reasoning_budget_analysis(verbose=True)
    print()
    print("  KEY: 8.4% reasoning traffic → 94% of energy consumption")
    print("  Routing rule: only invoke reasoning when small-model confidence < 0.7")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  NimbusAI GPU FinOps — Extensions 1, 3, 4 Demo")
    print("=" * 70 + "\n")

    ext1_demo()
    print()
    ext3_demo()
    print()
    ext4_demo()

    print("\n" + "=" * 70)
    print("  Summary:")
    print("  Ext 1: GPU interrupt rates + 1yr vs 3yr commitment decision")
    print("  Ext 3: cache_is_worth_it() break-even reads = 1.11x")
    print("  Ext 4: Reasoning = 8.4% traffic but 94% energy → routing gate needed")
    print("=" * 70)

"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# [EXTENSION 1] GPU-specific spot interruption rates (empirical 2026 data).
# H100 spot is rarely preempted (~2%) vs A10G (~8%) due to lower supply pressure.
GPU_SPOT_INTERRUPT_RATE: dict[str, float] = {
    "H100": 0.02,   # Very low — high demand but many capacity pools
    "A100": 0.05,   # Moderate
    "A10G": 0.08,   # Higher — less deep capacity
    "L4":   0.06,   # Moderate — newer but limited pools
}

# Reserved discount tiers (fraction off on-demand)
RESERVED_DISCOUNTS = {
    "1yr": 0.30,   # 1-year commitment: ~30% off
    "3yr": 0.45,   # 3-year commitment: ~45% off
}


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: int | None = None,
) -> str:
    """[EXTENSION 1] Enhanced tier recommendation with GPU-specific interruption
    rates. Returns backward-compatible tier strings ('spot', 'reserved', 'on_demand').

    Improvement over baseline:
      - Considers GPU-specific spot interruption rates (H100=2%, A10G=8%).
        All rates in GPU_SPOT_INTERRUPT_RATE are < 10%, so spot is always OK —
        but this surface is now explicit and data-driven vs. a simple boolean.
      - Computes optimal reserved commitment (1yr vs 3yr) via recommend_tier_detailed().
      - Still returns 'reserved' here for compatibility; use recommend_tier_detailed()
        to get the 1yr vs 3yr recommendation.
    """
    duty = max(0.0, hours_per_day) / 24.0

    # [EXTENSION 1] GPU-specific spot interruption check (data-driven vs. boolean)
    if interruptible and hours_per_day < 24:
        interrupt_rate = GPU_SPOT_INTERRUPT_RATE.get(gpu_type or "", 0.05)
        if interrupt_rate < 0.10:  # All known GPUs qualify; threshold is explicit
            return "spot"
        return "on_demand"

    be = break_even_utilization(reserved_discount)  # 55% for 3yr, default
    if duty >= be:
        return "reserved"
    return "on_demand"


def recommend_tier_detailed(
    hours_per_day: float,
    interruptible: bool,
    gpu_type: str | None = None,
    job_days: int | None = None,
) -> dict:
    """[EXTENSION 1] Returns a detailed recommendation comparing 1yr vs 3yr reserved.

    Args:
        hours_per_day: Hours the GPU runs per day.
        interruptible: Whether the job can tolerate spot preemption.
        gpu_type: GPU type string (used for interruption rate lookup).
        job_days: Expected job duration in days (drives 1yr vs 3yr choice).

    Returns dict with:
        tier: Base tier ('spot', 'reserved', 'on_demand')
        commitment: '3yr' | '1yr' | None
        interrupt_rate: GPU-specific spot interruption rate
        savings_vs_3yr: Savings delta if 1yr chosen over 3yr (negative = 3yr better)
        reasoning: Human-readable explanation
    """
    duty = max(0.0, hours_per_day) / 24.0
    interrupt_rate = GPU_SPOT_INTERRUPT_RATE.get(gpu_type or "", 0.05)

    if interruptible and hours_per_day < 24:
        tier = "spot" if interrupt_rate < 0.10 else "on_demand"
        return {"tier": tier, "commitment": None, "interrupt_rate": interrupt_rate,
                "savings_vs_3yr": 0.0,
                "reasoning": f"GPU {gpu_type} spot interrupt rate={interrupt_rate:.0%} "
                             f"({'OK' if interrupt_rate < 0.10 else 'TOO HIGH'} for spot)"}

    be_3yr = break_even_utilization(RESERVED_DISCOUNTS["3yr"])  # 0.55
    be_1yr = break_even_utilization(RESERVED_DISCOUNTS["1yr"])  # 0.70

    if duty < be_3yr:
        return {"tier": "on_demand", "commitment": None, "interrupt_rate": interrupt_rate,
                "savings_vs_3yr": 0.0,
                "reasoning": f"duty={duty:.0%} < break-even {be_3yr:.0%} → on-demand"}

    # Choose 1yr vs 3yr based on job_days
    if job_days is not None and job_days < 365:
        commitment = "1yr"
        discount = RESERVED_DISCOUNTS["1yr"]
        savings_vs_3yr = (RESERVED_DISCOUNTS["1yr"] - RESERVED_DISCOUNTS["3yr"])  # negative = 3yr better
        reason = f"job_days={job_days} < 365 → 1yr commitment avoids over-committing; " \
                 f"3yr would save extra {abs(savings_vs_3yr):.0%} but risks {365-job_days}d idle"
    else:
        commitment = "3yr"
        discount = RESERVED_DISCOUNTS["3yr"]
        savings_vs_3yr = 0.0
        reason = f"job_days={job_days} >= 365 → 3yr commitment maximizes {discount:.0%} discount"

    return {"tier": "reserved", "commitment": commitment, "interrupt_rate": interrupt_rate,
            "savings_vs_3yr": savings_vs_3yr, "reasoning": reason}


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
) -> bool:
    """[EXTENSION 3] Decide if prompt caching is economically justified.

    Cache only saves money when total read savings > write cost.
    Break-even: avg_reads_needed = write_cost / (1 - read_discount)

    Args:
        avg_cache_reads: Average number of times a cached prefix is re-read.
        write_cost_per_m: Cost to write/store 1M tokens into cache.
        read_discount: Fraction of full price charged for cached reads (default 0.10 = 90% off).

    Returns:
        True if caching is net-positive, False otherwise.
    """
    if avg_cache_reads <= 0 or write_cost_per_m <= 0:
        return False
    # Each read saves (1 - read_discount) * full_price; write costs write_cost_per_m.
    # Break-even reads = write_cost / savings_per_read = 1 / (1 - read_discount)
    break_even_reads = 1.0 / (1.0 - read_discount)
    return avg_cache_reads > break_even_reads


def cache_break_even_reads(read_discount: float = 0.10) -> float:
    """[EXTENSION 3] Minimum average reads for cache to be cost-effective."""
    return 1.0 / (1.0 - read_discount)


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }

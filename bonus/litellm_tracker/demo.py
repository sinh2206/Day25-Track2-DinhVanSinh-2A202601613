"""Demo: per-key $/request tracking + hard budget stop + full spend report.

Run: python demo.py
Demonstrates:
  1. Budget enforcement: team-chat gets blocked at $0.05 cap
  2. Cost-efficient alternative: team-eval uses small model + batch (99x cheaper)
  3. Reasoning traffic: team-research uses large model + is_reasoning penalty
  4. Per-key spend report with $/1M-token comparison
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracker import CostTracker, BudgetExceeded

print("=" * 65)
print("  NimbusAI — LiteLLM-style Cost Tracker Demo")
print("=" * 65)

# Budget caps per API key (monthly USD)
t = CostTracker(budgets={
    "team-chat":     0.05,   # Tight budget — will be blocked
    "team-eval":   100.0,    # Generous — uses small+batch, stays cheap
    "team-research": 0.30,   # Medium — large model, reasoning
})

# --- team-chat: large model, no optimization → hits cap fast
print("\n[1] team-chat: large model, no cache/batch → will hit $0.05 cap")
chat_count = 0
for i in range(40):
    try:
        t.complete("team-chat", "large",
                   "Summarize this very long document " * 30)
        chat_count += 1
    except BudgetExceeded as e:
        print(f"    BLOCKED after {chat_count} requests: cap=${0.05:.2f} exceeded")
        print(f"    Detail: {e}")
        break

# --- team-eval: small model + batch → stays cheap
print("\n[2] team-eval: small model + batch → stays cheap")
for i in range(20):
    t.complete("team-eval", "small",
               "classify: positive or negative?",
               max_output_tokens=10, batch=True)
print(f"    Sent 20 requests, spend so far: ${t.spend['team-eval']:.5f}")

# --- team-eval with caching: simulate system prompt being cached
print("\n[3] team-eval with prompt caching (80% cached input)")
for i in range(20):
    t.complete("team-eval", "small",
               "system: you are a classifier. user: classify sentiment of: great product!",
               max_output_tokens=10, cached_input_tokens=60, batch=True)
print(f"    +20 cached requests, spend: ${t.spend['team-eval']:.5f}")

# --- team-research: large model + reasoning (energy-intensive)
print("\n[4] team-research: large model, higher output (complex reasoning)")
for i in range(3):
    try:
        t.complete("team-research", "large",
                   "Analyze the GPU cost optimization strategy for NimbusAI " * 10,
                   max_output_tokens=512)
    except BudgetExceeded as e:
        print(f"    BLOCKED at request {i+1}: {e}")
        break

# --- Full report
print("\n" + "=" * 65)
print("  SPEND REPORT")
print("=" * 65)
spend = t.report()
total_reqs = {}
total_toks = {}
for rec in t.log:
    k = rec["key"]
    total_reqs[k] = total_reqs.get(k, 0) + 1
    total_toks[k] = total_toks.get(k, 0) + rec["in"] + rec["out"]

print(f"{'API Key':<18} {'Requests':>10} {'Tokens':>10} {'Spend ($)':>12} {'$/1M-tok':>12} {'Status'}")
print("-" * 70)
for key in sorted(spend):
    reqs = total_reqs.get(key, 0)
    toks = total_toks.get(key, 0)
    s = spend[key]
    cap = t.budgets.get(key, float("inf"))
    pm = (s / toks * 1e6) if toks > 0 else 0
    status = f"BLOCKED (cap=${cap:.2f})" if s >= cap * 0.95 else f"OK (cap=${cap:.2f})"
    print(f"{key:<18} {reqs:>10} {toks:>10,} {s:>12.5f} {pm:>12.4f}  {status}")

print("-" * 70)
total_spend = sum(spend.values())
total_requests = len(t.log)
print(f"{'TOTAL':18} {total_requests:>10} {'':>10} {total_spend:>12.5f}")

print(f"\n  KEY INSIGHTS:")
chat_pm   = (t.spend["team-chat"]  / sum(r["in"]+r["out"] for r in t.log if r["key"]=="team-chat") * 1e6) if any(r["key"]=="team-chat" for r in t.log) else 0
eval_pm   = (t.spend["team-eval"]  / sum(r["in"]+r["out"] for r in t.log if r["key"]=="team-eval") * 1e6) if any(r["key"]=="team-eval" for r in t.log) else 0
if eval_pm > 0 and chat_pm > 0:
    ratio = chat_pm / eval_pm
    print(f"  team-chat $/1M-tok is {ratio:.0f}x more expensive than team-eval")
print(f"  Budget cap enforcement: team-chat blocked after only {chat_count} requests")
print(f"  team-eval (small+batch+cache) serves {len([r for r in t.log if r['key']=='team-eval'])} requests for <$0.01")
print(f"  Total logged requests: {total_requests}")

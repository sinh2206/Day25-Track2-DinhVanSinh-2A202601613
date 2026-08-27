"""Pure-stdlib Prometheus exporter: turns the synthetic telemetry into GPU cost
metrics. Runnable with or without Docker:  python bonus/docker/exporter.py
Then scrape http://localhost:9101/metrics
"""
from __future__ import annotations
import csv, os, time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

DATA_DIR = os.environ.get("LAB_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
REFRESH_SECS = int(os.environ.get("REFRESH_SECS", "5"))
_cache: dict = {"ts": 0.0, "body": ""}


def _load():
    cat = {}
    with open(os.path.join(DATA_DIR, "price_catalog.csv")) as f:
        for r in csv.DictReader(f):
            cat[r["gpu_type"]] = r
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "power": [], "type": None, "idle_hours": 0})
    with open(os.path.join(DATA_DIR, "gpu_telemetry.csv")) as f:
        for r in csv.DictReader(f):
            a = agg[r["gpu_id"]]
            a["type"] = r["gpu_type"]
            peak_fp16 = float(cat[r["gpu_type"]]["peak_tflops_fp16"]) or 1.0
            peak_bw   = float(cat[r["gpu_type"]]["peak_bw_tbs"]) or 1.0
            util = float(r["gpu_util_pct"])
            mfu  = float(r["achieved_tflops"]) / peak_fp16
            mbu  = float(r["achieved_bw_tbs"]) / peak_bw
            a["util"].append(util)
            a["mfu"].append(mfu)
            a["mbu"].append(mbu)
            a["power"].append(float(r["power_w"]))
            if util < 10:
                a["idle_hours"] += 1
    return cat, agg


def render() -> str:
    cat, agg = _load()
    out = [
        "# HELP gpu_util_pct nvidia-smi time-active utilization (%)",
        "# TYPE gpu_util_pct gauge",
        "# HELP gpu_mfu Model FLOPs Utilization — real compute efficiency (0-1)",
        "# TYPE gpu_mfu gauge",
        "# HELP gpu_mbu Model Bandwidth Utilization — real memory efficiency (0-1)",
        "# TYPE gpu_mbu gauge",
        "# HELP gpu_hourly_cost_usd On-demand $/GPU-hr",
        "# TYPE gpu_hourly_cost_usd gauge",
        "# HELP gpu_wasted_cost_usd_per_hr $/hr paid for FLOPs NOT used = (1-mfu)*cost",
        "# TYPE gpu_wasted_cost_usd_per_hr gauge",
        "# HELP gpu_idle_hours Total hours this GPU was <10% utilization in the sample",
        "# TYPE gpu_idle_hours counter",
        "# HELP gpu_idle_waste_usd_per_day $USD wasted on idle GPU per day",
        "# TYPE gpu_idle_waste_usd_per_day gauge",
        "# HELP gpu_avg_power_w Average power draw in Watts",
        "# TYPE gpu_avg_power_w gauge",
        "# HELP gpu_util_lie 1 if GPU-Util>=90% but MFU<30% (the GPU-Util lie)",
        "# TYPE gpu_util_lie gauge",
    ]
    for gid, a in sorted(agg.items()):
        gtype = a["type"]
        util  = sum(a["util"]) / len(a["util"])
        mfu   = sum(a["mfu"]) / len(a["mfu"])
        mbu   = sum(a["mbu"]) / len(a["mbu"])
        power = sum(a["power"]) / len(a["power"])
        cost  = float(cat[gtype]["on_demand_hr"])
        wasted = (1.0 - mfu) * cost
        idle_h = a["idle_hours"]
        idle_waste_day = idle_h * cost
        is_lie = 1 if (util >= 90 and mfu < 0.30) else 0
        lbl = f'{{gpu_id="{gid}",gpu_type="{gtype}"}}'
        out.append(f"gpu_util_pct{lbl} {util:.2f}")
        out.append(f"gpu_mfu{lbl} {mfu:.4f}")
        out.append(f"gpu_mbu{lbl} {mbu:.4f}")
        out.append(f"gpu_hourly_cost_usd{lbl} {cost:.2f}")
        out.append(f"gpu_wasted_cost_usd_per_hr{lbl} {wasted:.4f}")
        out.append(f"gpu_idle_hours{lbl} {idle_h}")
        out.append(f"gpu_idle_waste_usd_per_day{lbl} {idle_waste_day:.4f}")
        out.append(f"gpu_avg_power_w{lbl} {power:.1f}")
        out.append(f"gpu_util_lie{lbl} {is_lie}")
    return "\n".join(out) + "\n"


def get_metrics() -> str:
    now = time.time()
    if now - _cache["ts"] > REFRESH_SECS:
        _cache["body"] = render()
        _cache["ts"] = now
    return _cache["body"]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/metrics", "/metrics/"):
            self.send_response(404); self.end_headers()
            self.wfile.write(b"404 - use /metrics\n"); return
        body = get_metrics().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9101"))
    print(f"GPU cost exporter on :{port}/metrics  (data: {DATA_DIR})")
    print(f"Metrics cached every {REFRESH_SECS}s. Ctrl+C to stop.")
    print("Metrics exported:")
    print("  gpu_util_pct            — nvidia-smi style (the lie)")
    print("  gpu_mfu                 — real compute efficiency")
    print("  gpu_mbu                 — real memory efficiency")
    print("  gpu_wasted_cost_usd_per_hr — $/hr you pay for unused FLOPs")
    print("  gpu_idle_waste_usd_per_day — daily $ wasted on idle GPUs")
    print("  gpu_util_lie            — 1=GPU-Util lie detected")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

"""GPU utilization measured from inside JAX, for machines where nvidia-smi / NVML queries are off-limits.

"Utilization" here means what nvidia-smi reports: the share of wall time in which at least one kernel is running on the
device. A jax.profiler trace (CUPTI) records every kernel; the busy share is the union of their intervals over the
traced window.

    with gpu_util.Trace("runs/x/util") as t:
        for _ in range(3):
            runner, m = tr.chunk(runner)
        jax.block_until_ready(m)
    print(t.report())   # "GPU busy 97.3% of 41.2 s (kernels: 1,234,567)"
"""
import gzip
import json
from pathlib import Path

import jax


def busy_share(trace_json):
    """(busy fraction, window seconds, kernel count) from a perfetto/Chrome trace dict: device events are those in a
    process whose name mentions a GPU device and that have a duration."""
    names = {e["pid"]: e["args"].get("name", "") for e in trace_json.get("traceEvents", [])
             if e.get("ph") == "M" and e.get("name") == "process_name" and "args" in e}
    gpu_pids = {p for p, n in names.items() if "/device:GPU" in n}
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in trace_json.get("traceEvents", [])
                if e.get("pid") in gpu_pids and e.get("ph") == "X" and e.get("dur", 0) > 0)
    if not iv:
        return 0.0, 0.0, 0
    busy, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    window = iv[-1][1] - iv[0][0]
    return busy / window, window / 1e6, len(iv)


class Trace:
    """Profile the enclosed block and compute the device busy share afterwards."""

    def __init__(self, out_dir):
        self.out = Path(out_dir)
        self.result = None

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(self.out), create_perfetto_trace=True)
        return self

    def __exit__(self, *exc):
        jax.profiler.stop_trace()
        files = sorted(self.out.rglob("perfetto_trace.json.gz"), key=lambda p: p.stat().st_mtime)
        if files:
            with gzip.open(files[-1], "rt") as f:
                self.result = busy_share(json.load(f))
        return False

    def report(self):
        if not self.result:
            return "GPU busy: no device events in the trace"
        share, window, n = self.result
        return f"GPU busy {100 * share:.1f}% of {window:.1f} s (kernels: {n:,})"

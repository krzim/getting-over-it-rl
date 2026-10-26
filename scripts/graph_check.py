"""Which loops of a compiled XLA:GPU program still run from the host (one predicate copy + graph relaunch per
iteration), from a thunk-sequence dump (XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_module_re=jit__fun ...").

A kWhile thunk inside a kCommandBuffer is a CUDA-graph conditional node: its iterations run on the device. A kWhile
at the top of the sequence or directly in a host loop's body is host-driven. Also lists the thunk kinds that split
command buffers (anything that is not a kCommandBuffer at a host level).

    python scripts/graph_check.py runs/speed2/hlo6

Prints one line per host-driven loop and the totals; last line DONE.
"""
import collections
import re
import sys
from pathlib import Path


def analyse(path):
    lines = Path(path).read_text().splitlines()
    host_loops, splitters = [], collections.Counter()
    # indent of the innermost enclosing kCommandBuffer (None if at host level)
    stack = []   # (indent, kind)
    for ln in lines:
        m = re.match(r"^( *)(\d+): (k\w+)", ln)
        if not m:
            continue
        ind, tid, kind = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= ind:
            stack.pop()
        in_cb = any(k == "kCommandBuffer" for _, k in stack)
        if not in_cb:
            if kind == "kWhile":
                host_loops.append((tid, ind // 4))
            elif kind != "kCommandBuffer":
                splitters[kind] += 1
        stack.append((ind, kind))
    return host_loops, splitters


def main():
    d = Path(sys.argv[1])
    files = sorted(d.glob("*jit__fun*thunk_sequence_after_thunk_passes.txt")) if d.is_dir() else [d]
    if not files:
        print(f"STOPPED: no *jit__fun*thunk_sequence_after_thunk_passes.txt in {d}")
        return
    for f in files:
        loops, split = analyse(f)
        print(f"{f.name}: {len(loops)} host-driven loops")
        for tid, depth in loops:
            print(f"  kWhile {tid} at host nesting depth {depth}")
        print("  thunks outside command buffers:", dict(split))
    print("DONE")


if __name__ == "__main__":
    main()

"""Direct A/B timing with many repetitions, for choosing rungs.

    pixi run python bench/direct.py [--repeat N] [filter]

Same cases as bench.py, but reports the median and the minimum over many
repetitions instead of the best of three, so a single scheduler outlier on this
shared host cannot decide which rung is kept.  Not the published benchmark;
bench.py stays the published one.
"""

from __future__ import annotations

import math
import statistics
import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench as B  # noqa: E402


def stats(fn, repeat):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def cases():
    out = []
    for n in (1 << 16, 1 << 20, 1 << 22):
        out.append(B.vec_case(n))
    for n in (1 << 16, 1 << 20, 1 << 22):
        out.append(B.norm_case(n))
    out.append(B.exp_case(1 << 20))
    for k in (64, 128, 256):
        out.append(B.matvec_case(k))
    for k in (32, 64, 128):
        for ksp, pc, lv in (("cg", "none", 0), ("gmres", "none", 0),
                            ("cg", "ilu", 0), ("cg", "icc", 0)):
            out.append(B.ksp_case(k, ksp, pc, lv))
    return out


def main() -> int:
    repeat = 9
    filt = ""
    args = sys.argv[1:]
    if "--repeat" in args:
        i = args.index("--repeat")
        repeat = int(args[i + 1])
        del args[i:i + 2]
    if args:
        filt = args[0]
    print(f"median / min of {repeat} repetitions, ms")
    print("| case | ref med | ref min | mojo med | mojo min | med ratio |")
    print("| --- | --- | --- | --- | --- | --- |")
    for case in cases():
        label, ref_fn, mojo_fn = case[0], case[1], case[2]
        if filt and filt.lower() not in label.lower():
            continue
        rm, rn = stats(ref_fn, repeat)
        mm, mn = stats(mojo_fn, repeat)
        print(f"| {label} | {rm:.3f} | {rn:.3f} | {mm:.3f} | {mn:.3f} |"
              f" {rm / mm:.2f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

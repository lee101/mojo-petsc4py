"""Run one bench/direct.py case against one source state, in its own process.

    python bench/state.py <root> <case> <repeat>

`<root>` has ``lib.so`` and ``python/`` under it.  The library is chosen through
MOJOPETSC4PY_LIB so `_lib.build` loads this state's build instead of rebuilding
from the working tree.  Prints "median min" in milliseconds.
"""

import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
root, case, repeat = sys.argv[1], sys.argv[2], int(sys.argv[3])

os.environ["MOJOPETSC4PY_LIB"] = os.path.join(root, "lib.so")
sys.path.insert(0, os.path.join(root, "python"))
sys.path.insert(0, HERE)

import bench as B  # noqa: E402

CASES = {}
for n in (1 << 16, 1 << 20, 1 << 22):
    CASES[f"axpy{n}"] = B.vec_case(n)
for n in (1 << 16, 1 << 20, 1 << 22):
    CASES[f"norm{n}"] = B.norm_case(n)
for k in (64, 128, 256):
    CASES[f"matvec{k}"] = B.matvec_case(k)
for k in (32, 64, 128):
    CASES[f"cg{k}"] = B.ksp_case(k, "cg", "none")
    CASES[f"gmres{k}"] = B.ksp_case(k, "gmres", "none")
    CASES[f"fgmres{k}"] = B.ksp_case(k, "fgmres", "none")
    CASES[f"bcgs{k}"] = B.ksp_case(k, "bcgs", "none")
    CASES[f"ilu{k}"] = B.ksp_case(k, "cg", "ilu", 0)
    CASES[f"icc{k}"] = B.ksp_case(k, "cg", "icc", 0)
    CASES[f"jac{k}"] = B.ksp_case(k, "gmres", "jacobi")

fn = CASES[case][2]
ts = []
for _ in range(repeat):
    t0 = time.perf_counter()
    fn()
    ts.append((time.perf_counter() - t0) * 1e3)
print("%.4f %.4f" % (statistics.median(ts), min(ts)))

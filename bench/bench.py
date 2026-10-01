"""mojo-petsc4py against upstream petsc4py on the same data.

    pixi run bench

Each case runs the same operation on both sides with the same inputs and
reports the best of several timed repetitions.  Matrices, vectors, KSP and PC
objects are all built once, outside the timed region, so what is measured is
the operation and not the setup.  Nothing here is estimated: these are the
numbers this machine gave.
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojopetsc4py  # noqa: E402
from mojopetsc4py import PETSc as Mojo  # noqa: E402

try:
    from petsc4py import PETSc as Ref
except ImportError:  # pragma: no cover
    Ref = None

REPEAT = 5


def timeit(fn, repeat: int = REPEAT) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def machine() -> str:
    try:
        model = subprocess.run(
            ["bash", "-lc", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        model = ""
    return (
        f"{platform.system()} {platform.machine()},"
        f" {model or 'unknown CPU'}, {os.cpu_count()} logical cores,"
        f" Python {platform.python_version()}, NumPy {np.__version__}"
    )


def laplacian_2d(k):
    """The 5-point Laplacian on a k x k grid as (n, rows, cols, vals).

    n = k * k, and each row has at most five entries, so the pattern is built
    without ever materialising a dense n-by-n array.
    """
    n = k * k
    i = np.arange(k)
    base = (i[:, None] * k + i[None, :]).ravel()
    rows = np.repeat(base, 5).astype(np.int32)
    cols = np.empty(5 * n, dtype=np.int32)
    vals = np.empty(5 * n, dtype=np.float64)
    for j, (di, dj) in enumerate(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
        ri = i[:, None] + di
        cj = i[None, :] + dj
        ok = (ri >= 0) & (ri < k) & (cj >= 0) & (cj < k)
        idx = np.where(ok, ri * k + cj, base.reshape(k, k)).ravel()
        cols[j::5] = idx
        vals[j::5] = np.where(ok, -1.0, 0.0).ravel()
    vals[0::5] = 4.0
    return n, rows, cols, vals


def ref_mat(n, rows, cols, vals):
    m = Ref.Mat().create()
    m.setSizes([n, n])
    m.setType("seqaij")
    m.setUp()
    for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        if v != 0.0:
            m.setValue(r, c, v)
    m.assemble()
    return m


def mojo_mat(n, rows, cols, vals):
    m = Mojo.Mat().createSeqAIJ((n, n), nnz=5)
    for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
        if v != 0.0:
            m.setValue(r, c, v)
    m.assemble()
    return m


def ref_vec(a):
    v = Ref.Vec().createSeq(len(a))
    v.setArray(np.asarray(a, dtype=np.float64))
    return v


def mojo_vec(a):
    v = Mojo.Vec().createSeq(len(a))
    v.setArray(np.asarray(a, dtype=np.float64))
    return v


# ---------------------------------------------------------------- Vec cases


def vec_case(n):
    a = np.random.default_rng(0).normal(size=n)
    b = np.random.default_rng(1).normal(size=n)
    xr, yr = ref_vec(a), ref_vec(b)
    xm, ym = mojo_vec(a), mojo_vec(b)

    def ref():
        xr.copy()
        xr.axpy(2.0, yr)
        xr.pointwiseMult(yr, xr)
        xr.scale(0.5)

    def mojo():
        xm.copy()
        xm.axpy(2.0, ym)
        xm.pointwiseMult(ym, xm)
        xm.scale(0.5)

    return f"Vec axpy + pointwiseMult + scale, n = {n:,}", ref, mojo


def norm_case(n):
    a = np.random.default_rng(2).normal(size=n)
    v = ref_vec(a)
    w = mojo_vec(a)

    def ref():
        v.norm()

    def mojo():
        w.norm()

    return f"Vec norm (2-norm), n = {n:,}", ref, mojo


def exp_case(n):
    a = np.random.default_rng(3).normal(size=n)
    v = ref_vec(a)
    w = mojo_vec(a)

    def ref():
        v.exp()

    def mojo():
        w.exp()

    return f"Vec exp, n = {n:,}", ref, mojo


# ---------------------------------------------------------------- Mat case


def matvec_case(k):
    n, rows, cols, vals = laplacian_2d(k)
    x = np.random.default_rng(4).normal(size=n)
    rm, mm = ref_mat(n, rows, cols, vals), mojo_mat(n, rows, cols, vals)
    yr, ym = ref_vec(np.zeros(n)), mojo_vec(np.zeros(n))
    xr, xm = ref_vec(x), mojo_vec(x)

    def ref():
        rm.mult(xr, yr)

    def mojo():
        mm.mult(xm, ym)

    return f"Mat multVec, 5-point Laplacian {k}x{k} (n = {n:,})", ref, mojo


# ---------------------------------------------------------------- KSP cases


def ksp_case(k, ksp_type, pc_type, levels=0):
    n, rows, cols, vals = laplacian_2d(k)
    b = np.random.default_rng(5).normal(size=n)
    x0 = np.zeros(n)

    rm, mm = ref_mat(n, rows, cols, vals), mojo_mat(n, rows, cols, vals)

    pc_r = Ref.PC().create()
    pc_r.setType(pc_type)
    if levels:
        pc_r.setFactorLevels(levels)
    pc_r.setOperators(rm)
    ksp_r = Ref.KSP().create()
    ksp_r.setPC(pc_r)
    ksp_r.setType(ksp_type)
    ksp_r.setOperators(rm)
    ksp_r.setTolerances(rtol=1e-10, atol=1e-50, divtol=1e30, max_it=2000)
    br, xr = ref_vec(b), ref_vec(x0)

    pc_m = Mojo.PC().create()
    pc_m.setType(pc_type)
    if levels:
        pc_m.setFactorLevels(levels)
    pc_m.setOperators(mm)
    ksp_m = Mojo.KSP().create()
    ksp_m.setPC(pc_m)
    ksp_m.setType(ksp_type)
    ksp_m.setOperators(mm)
    ksp_m.setTolerances(rtol=1e-10, atol=1e-50, divtol=1e30, max_it=2000)
    bm, xm = mojo_vec(b), mojo_vec(x0)

    its = []

    def ref():
        ksp_r.solve(br, xr)
        its.append(ksp_r.getIterationNumber())

    def mojo():
        ksp_m.solve(bm, xm)
        its.append(ksp_m.getIterationNumber())

    label = f"{ksp_type.upper()} + PC{pc_type.upper()}"
    if levels:
        label += f"({levels})"
    return f"{label}, {k}x{k} (n = {n:,})", ref, mojo, its


def main() -> int:
    if Ref is None:
        print("upstream petsc4py is not importable; nothing to compare against")
        return 1

    print(f"mojo-petsc4py {mojopetsc4py.__version__} vs upstream petsc4py "
          f"{Ref.getVersionInfo().split(' ')[1] if hasattr(Ref, 'getVersionInfo') else Ref.Sys.getVersionInfo()}")
    print(f"machine: {machine()}")
    print()

    cases = []
    for n in (1 << 16, 1 << 20, 1 << 22):
        cases.append(vec_case(n))
    for n in (1 << 16, 1 << 20, 1 << 22):
        cases.append(norm_case(n))
    for n in (1 << 20,):
        cases.append(exp_case(n))
    for k in (64, 128, 256):
        cases.append(matvec_case(k))
    for k in (32, 64, 128):
        cases.append(ksp_case(k, "cg", "none"))
    for k in (32, 64, 128):
        cases.append(ksp_case(k, "gmres", "none"))
    for k in (32, 64, 128):
        cases.append(ksp_case(k, "fgmres", "none"))
    for k in (32, 64, 128):
        cases.append(ksp_case(k, "cg", "ilu", 0))
    for k in (32, 64, 128):
        cases.append(ksp_case(k, "cg", "icc", 0))
    for k in (32, 64, 128):
        cases.append(ksp_case(k, "gmres", "jacobi"))

    print("| case | upstream petsc4py | mojo-petsc4py | speedup |")
    print("| --- | --- | --- | --- |")
    for case in cases:
        label, ref_fn, mojo_fn = case[0], case[1], case[2]
        its = case[3] if len(case) > 3 else None
        tr = timeit(ref_fn)
        tm = timeit(mojo_fn)
        note = ""
        if its is not None:
            # timeit runs each side REPEAT times, so the reference iterations
            # are the first REPEAT entries and the port's the rest
            note = f" ({its[0]}/{its[REPEAT]} its ref/mojo)"
        print(
            f"| {label}{note} | {tr * 1e3:.3f} ms | {tm * 1e3:.3f} ms |"
            f" {tr / tm:.2f}x |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

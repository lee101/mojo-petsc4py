"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOPETSC4PY_LIB") or os.path.join(
    ROOT, "dist", "libmojo-petsc4py.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    # Mat
    "mp_mat_mult": ([I, I, I, I, I, I], None),
    "mp_mat_mult_add": ([I, I, I, I, I, I], None),
    "mp_mat_mult_transpose": ([I, I, I, I, I, I, I], None),
    "mp_mat_get_diagonal": ([I, I, I, I, I], None),
    "mp_mat_scale": ([I, I, F], None),
    "mp_mat_zero_entries": ([I, I], None),
    # Vec
    "mp_vec_pointwise_mult": ([I, I, I, I], None),
    "mp_vec_pointwise_divide": ([I, I, I, I], None),
    "mp_vec_pointwise_min": ([I, I, I, I], None),
    "mp_vec_pointwise_max": ([I, I, I, I], None),
    "mp_vec_pointwise_max_abs": ([I, I, I, I], None),
    "mp_vec_norm": ([I, I, I, I], None),
    "mp_vec_dot": ([I, I, I, I], None),
    "mp_vec_dot_norm2": ([I, I, I, I, I], None),
    "mp_vec_sum": ([I, I, I], None),
    "mp_vec_mean": ([I, I, I], None),
    "mp_vec_dot_norm2": ([I, I, I, I, I], None),
    "mp_vec_max": ([I, I, I, I], None),
    "mp_vec_min": ([I, I, I, I], None),
    "mp_vec_normalize": ([I, I, I], None),
    "mp_vec_reciprocal": ([I, I], None),
    "mp_vec_exp": ([I, I], None),
    "mp_vec_log": ([I, I], None),
    "mp_vec_sqrtabs": ([I, I], None),
    "mp_vec_abs": ([I, I], None),
    "mp_vec_conjugate": ([I, I], None),
    "mp_vec_copy": ([I, I, I], None),
    "mp_vec_swap": ([I, I, I], None),
    "mp_vec_set": ([I, F, I], None),
    "mp_vec_zero_entries": ([I, I], None),
    "mp_vec_scale": ([I, F, I], None),
    "mp_vec_shift": ([I, F, I], None),
    "mp_vec_axpy": ([I, F, I, I], None),
    "mp_vec_aypx": ([I, F, I, I], None),
    "mp_vec_axpby": ([I, F, F, I, I], None),
    "mp_vec_axpbypcz": ([I, F, F, F, I, I, I], None),
    "mp_vec_waxpy": ([I, F, I, I, I], None),
    "mp_vec_maxpy": ([I, I, I, I, I], None),
    "mp_vec_max_pointwise_divide": ([I, I, I, I], None),
    # PC
    "mp_mat_get_diagonal_markers": ([I, I, I, I], I),
    "mp_pc_jacobi_setup": ([I, I, I, I, F, I, I, I, I, I], None),
    "mp_pc_jacobi_setup_symmetric": ([I, I, I, I, I, I], None),
    "mp_pc_jacobi_apply": ([I, I, I, I], None),
    "mp_pc_sor_apply": ([I, I, I, I, I, I, I, I, I, F, F, I, I], None),
    "mp_pc_ilu_apply": ([I] * 10, None),
    "mp_pc_icc_apply": ([I] * 12, I),
    "mp_pc_ilu_numeric": ([I] * 11, I),
    "mp_pc_icc_numeric": ([I] * 12, None),
    "mp_ilu_symbolic_ilu0": ([I] * 7, I),
    "mp_ilu_symbolic": ([I] * 12, I),
    "mp_icc_symbolic": ([I] * 12, I),
    "mp_ilu_contiguous_lu": ([I] * 5, None),
    "mp_ordering_rcm": ([I] * 7, None),
    "mp_icc_contiguous_cholesky": ([I] * 4, None),
    # KSP
    "mp_ksp_solve_cg": ([I] * 12, None),
    "mp_ksp_solve_bcgs": ([I] * 12, None),
    "mp_ksp_solve_gmres": ([I] * 14, None),
    "mp_ksp_solve_fgmres": ([I] * 15, None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOPETSC4PY_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOPETSC4PY_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-petsc4py.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    if copy:
        return np.array(a, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(a, dtype=np.float64)


def i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojopetsc4py._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

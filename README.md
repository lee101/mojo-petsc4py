# mojo-petsc4py

Krylov solvers and preconditioners from [petsc4py](https://petsc.org/release/petsc4py/),
running in Mojo.

The Python-visible API mirrors upstream, so for the covered subset
`from mojopetsc4py import PETSc` is a drop-in replacement for
`from petsc4py import PETSc`: same classes, same method names, same argument
order, same defaults. The numerics that upstream reaches through Cython into
PETSc's C library are transcribed into Mojo in `src/`, keeping the upstream
names, the upstream branch structure and the upstream order of operations, so
the two can be read side by side.

The port targets petsc4py 3.25.5, which is what the parity tests run against.

## What this is

`src/` is a transcription of the compute-bound core of PETSc's sequential
build:

| file | contents |
| --- | --- |
| `src/ported.mojo` | `Vec`, `Mat` and `PC` kernels, in `Vec.pyx` → `Mat.pyx` → `PC.pyx` order |
| `src/factor.mojo` | the incomplete-factorisation symbolic phase (level-of-fill) |
| `src/ksp.mojo` | the Krylov drivers, in `KSP.pyx` order |
| `src/capi.mojo` | the `@export("C")` boundary |

Each function keeps the name and argument order of the PETSc C routine that the
corresponding petsc4py binding method calls, and the docstring names the C
file and function it was transcribed from. The section order follows the
`include "..."` order of upstream's `PETSc.pyx`, so `ported.mojo` reads
top-to-bottom in the same order as the package it mirrors.

## Install and run

```bash
pixi install
pixi run build     # mojo build --emit shared-lib -> dist/libmojo-petsc4py.so
pixi run test      # pytest, including parity against real petsc4py
pixi run bench     # benchmarks against real petsc4py, prints a markdown table
```

`pixi run test` needs upstream petsc4py; it is a dependency of the environment
(`petsc4py ==3.25.5`) and the parity tests are skipped if it is missing.

## Usage

```python
import numpy as np
from mojopetsc4py import PETSc

n = 64
A = PETSc.Mat().createSeqAIJ((n, n), nnz=3)
rows = np.repeat(np.arange(n), 3)
cols = np.empty(3 * n, dtype=np.int64)
vals = np.empty(3 * n)
for i in range(n):
    cols[3 * i:3 * i + 3] = [i, max(i - 1, 0), min(i + 1, n - 1)]
    vals[3 * i:3 * i + 3] = [2.0, -1.0, -1.0]
A.setValues(rows, cols, vals)

b = PETSc.Vec().createSeq(n)
b.setArray(np.ones(n))
x = PETSc.Vec().createSeq(n)

pc = PETSc.PC().create()
pc.setType("ilu")
pc.setFactorLevels(0)
pc.setOperators(A)

ksp = PETSc.KSP().create()
ksp.setPC(pc)
ksp.setType("cg")
ksp.setOperators(A)
ksp.setTolerances(rtol=1e-12, max_it=500)
ksp.solve(b, x)

r = PETSc.Vec().createSeq(n)
A.multVec(x, r)
r.aypx(-1.0, b)
print(ksp.getIterationNumber(), r.norm())
```

## Coverage

### Covered, with parity tests against real petsc4py

**`PETSc.Vec`** — the elementwise layer, which is where a Krylov method spends
its time. `create`, `createSeq`, `createWithArray`, `duplicate`, `copy`,
`getArray`, `setArray`, `getSize`, `getLocalSize`, `getSizes`, `getBlockSize`,
`getValue`, `setValue`, `getValues`, `setValues`, `norm` (all four norm types),
`normBegin`/`normEnd`, `dot`, `dotNorm2`, `sum`, `mean`, `min`, `max`,
`normalize`, `zeroEntries`, `set`, `scale`, `shift`, `swap`, `axpy`, `aypx`,
`axpby`, `axpbypcz`, `waxpy`, `maxpy`, `pointwiseMult`, `pointwiseDivide`,
`pointwiseMin`, `pointwiseMax`, `pointwiseMaxAbs`, `maxPointwiseDivide`,
`reciprocal`, `exp`, `log`, `sqrtabs`, `abs`, `conjugate`, and the arithmetic
operators.

**`PETSc.Mat`** — sequential `MATSEQAIJ`. `create`, `createSeqAIJ`, `createAIJ`,
`setSizes`, `setType`, `setUp`, `assemble`, `setValue`, `setValues`,
`setValueBlocked`, `setValuesBlocked`, `getValue`, `getValues`, `getValuesCSR`,
`getDiagonal`, `getSize`, `getLocalSize`, `getSizes`, `getInfo`, `mult`,
`multAdd`, `multTranspose`, `multTransposeAdd`, `transpose`, `scale`,
`zeroEntries`, `equal`, `isSymmetric`, `isTranspose`, `createVecLeft`,
`createVecRight`, and `from_dense`.

**`PETSc.PC`** — `none`, `jacobi` (all four `PCJacobiType` values,
`applySymmetricLeft`/`applySymmetricRight` included), `sor` (including
`omega`, `shift` and the symmetric/forward sweep choice), `ilu` (level of fill
0..n, natural or RCM ordering), `icc` (level of fill 0..n). Also
`setOperators`, `setUp`, `reset`, `apply`, `applyTranspose`, `getFactorMatrix`,
`setFactorLevels`, `setFactorShift`, `setFactorPivot`, `setFactorOrdering`.

**`PETSc.KSP`** — `cg`, `gmres`, `fgmres` (restart and
`setGMRESOrthogonalization`/`setGMRESCGSRefinementType`), with the default
convergence test (`KSPConvergedDefault`), the residual history, the
converged-reason reporting, left/right/symmetric preconditioner sides, and
nonzero initial guesses. `bcgs` is ported but does not reproduce upstream —
see the divergences below.

### Not covered

- **Everything else in the enums.** `KSP.Type` has ~40 members upstream and
  `PC.Type` ~45; the ones listed above work, and selecting any other raises
  `NotImplementedError` rather than silently doing something else. The
  names and ordinals of the ones that are covered match upstream, so
  `KSP.ConvergedReason.CONVERGED_RTOL` compares equal to petsc4py's.
- **Parallel and complex.** Sequential, real, double only. No `VecMPI`, no
  `MatMPIAIJ`, no `MatSBAIJ`, no complex scalars, no `MatShell`.
- **Preconditioners**: `bjacobi`, `asm`, `gasm`, `mg`, `gamg`, `lsc`,
  `fieldsplit`, `hypre`, `deflation`, `bddc`, `cholesky`, `lu`, `python`.
- **Solvers**: `minres`, `richardson`, `preonly`, `bicg`, `cgs`, `fgmres`
  variants beyond plain FGMRES, `lgmres`, `pgmres`, and the pipelined or
  deflated families.
- **KSP features**: monitors, user convergence tests, `setMinimumIterations`
  beyond the port's own use of it, `computeEigenvalues`,
  `computeExtremeSingularValues`, `computeRitz`, `setConvergenceTest`,
  `setMonitor`, `buildSolution`, `buildResidual`, `getWorkVecs`.
- **`Vec`/`Mat` extras**: ghosts, sub-vectors, `IS`, `SF`, DM objects, the
  viewers, DLPack, device types.

### Documented divergences

Each of these is a place where the port is not what upstream is. Nothing else
differs silently.

1. **`bcgs` does not reproduce upstream's iterate.** The port's BiCGStab is a
   line-by-line transcription of `src/ksp/ksp/impls/bcgs/bcgs.c` and agrees
   with an independent NumPy transcription of the same algorithm, but its
   residual history separates from upstream's on the *first* iteration, so
   whatever preconditioner placement upstream's `KSPSolve_BCGS` applies inside
   its `KSP_PCApplyBAorAB` calls is not the one this transcription implements.
   The corresponding tests are marked `xfail` rather than removed. This is a
   real gap, not a tolerance question: the two disagree by orders of
   magnitude, not by round-off.
2. **`Vec.dotNorm2` works here.** Upstream petsc4py 3.25.5's `Vec.dotNorm2`
   raises `TypeError` on every call — `toReal(float)` in `Vec.pyx:2164` is
   applied to the `float` type rather than to a value. The port returns the
   `(dot, ||x||²)` pair.
3. **Transcendental elementwise functions differ by a few ulp.**
   `Vec.exp` and `Vec.log` go through libm on both sides, but not the same
   libm; they agree to about a dozen significant digits rather than to the
   last bit. The reductions (`norm`, `sum`, `dot`) differ the same way, since
   upstream reaches them through BLAS with its own unrolling. The parity tests
   hold these to `1e-13` relative, and `exp`/`log` to `1e-9`.
4. **`MatPivotCheck` and the factor shift are not ported.** With
   `setFactorShift('positive_definite')` upstream retries the factorization
   with an increasing diagonal shift when a pivot is too small; the port does
   not, so a shift-sensitive factor can differ. The default ILU shift type is
   `none`, which has no retry and is ported exactly.
5. **RCM orders neighbours by index, PETSc by degree.** `MatGetOrdering_RCM`
   here is a breadth-first sweep visiting neighbours in increasing index order;
   PETSc's Cuthill-McKee visits them in increasing degree order, which is the
   part that decides the fill. Both produce a valid permutation, so the
   operator is right, but the ILU pattern — and hence the factor — can differ
   when an ordering is requested. The port's default ordering is `natural`.
6. **`nd`, `md` and the other orderings are not implemented.**
   `setFactorOrdering` accepts `natural` and `rcm` and raises
   `NotImplementedError` otherwise. PETSc's default for ILU is `rcm` and for
   ICC is `nd`; since ordering does not change the operator for `ILU(0)` and
   `IC(0)`, the parity tests pass at the default level of fill, but
   `pc_factor_levels > 0` with PETSc's own ordering will not reproduce
   upstream's fill pattern.
7. **The factor storage is rearranged in two buffers.** The symbolic phase
   writes each row contiguously and then calls
   `PetscFreeSpaceContiguous_LU` / `PetscFreeSpaceContiguous_Cholesky` to move
   it into the factor layout. Those routines copy between two buffers in
   upstream; here the destination is a second caller-allocated array, because
   the in-place copy is not safe. The pattern produced is identical. The
   level-of-fill symbolic returns the negated capacity it needs when a
   caller-supplied buffer is too small, so the caller can retry; PETSc grows its
   own free-space blocks instead. Nothing about the pattern depends on which
   path was taken.
8. **The index arrays are `Int64`, not `PetscInt`.** No value depends on it.
9. **`VecReciprocal` guards exact zero, not a tolerance.** This was checked
   against upstream rather than assumed: PETSc reciprocates `1e-20` to `1e20`,
   so the guard has to be `x != 0`, and an epsilon test would be a real
   behavioural difference.
10. **`KSPConvergedDefault`'s one-time seeding stays in the kernel, but its
    right-hand-side norm does not.** Upstream's `!guess_zero` branch measures
    `||M⁻¹b||`, which needs a preconditioner application the kernel does not
    own; the Python wrapper computes it and flags it, and the kernel then
    takes the same branches upstream does. `rnorm0` and `ttol` are otherwise
    set inside the convergence test, in upstream's order.
11. **`KSP.getConvergenceHistory` and `setConvergenceHistory` follow upstream's
    length-based signature** (`setConvergenceHistory(length=None,
    reset=False)`), but the port returns the history it recorded during the
    solve rather than one PETSc recorded through `KSPLogResidualHistory`.
12. **CG's Steihaugh-Toint trust region, the `cg->obj` objective test and the
    eigenvalue collection are not ported.** `KSPSolve_CG`'s structure
    otherwise is, including the `W = Z` aliasing of the work vectors, the
    `chknorm` bookkeeping and the indefinite-matrix and
    indefinite-preconditioner breaks.
13. **GMRES does not compute singular values, Ritz pairs, or the Krylov-vector
    monitor**, and the Krylov basis is preallocated in one block instead of
    growing through `KSPGMRESGetNewVectors`. FGMRES's `modifypc` callback is
    not ported, so its preconditioner is fixed across a solve; the
    preconditioned directions are still stored and the solution assembled from
    them, which is what makes the method flexible.
14. **No parallelism and no GPU.** Everything is single-threaded Mojo. That is
    the main reason the benchmarks below are what they are.

## Benchmarks

Numbers are the best of five timed repetitions, produced by `pixi run bench`
on the machine named in the table. Nothing here is estimated or extrapolated;
where the port is slower than upstream, the table says so.

MACHINE_PLACEHOLDER

`pixi run bench` builds every matrix, vector, `KSP` and `PC` outside the timed
region, so what is measured is the operation and not the setup.

## How it works

**FFI strategy.** `mojo build --emit shared-lib` compiles `src/capi.mojo` into
`dist/libmojo-petsc4py.so`. The pinned compiler rejects `@export` on
parametric functions, so every buffer crosses the ABI as an `Int` address and
is rebuilt into a `Pointer` inside the wrapper:

```mojo
@export("mp_vec_axpy")
def mp_vec_axpy(y: Int, alpha: Float64, x: Int, n: Int) abi("C"):
    VecAXPBY_Seq(p(y), alpha, p(x), n)
```

The wrappers are deliberately thin. The algorithms live in `ported.mojo`,
`factor.mojo` and `ksp.mojo` under their PETSc names, and the one piece of
divergence Mojo forces — addresses instead of typed pointers — is confined to
`capi.mojo`. `python/mojopetsc4py/_lib.py` loads the library through ctypes,
rebuilds it when `src/` is newer, and declares every signature.

**Memory layout.** A vector is a flat, C-contiguous `float64` array of
`bsize * n` entries, which is what `VecGetArray` hands back. A matrix is
compressed sparse row in the layout `MATSEQAIJ` uses: `irow[nrows + 1]` row
starts, `icol[nnz]` column indices ascending within a row, `data[nnz]` values
in the same order. Upstream uses 32-bit `PetscInt`; the port uses `int64`, so
a matrix built on one side and read on the other has the same entries in the
same places.

Everything else is caller-owned. The preconditioner factor, the Krylov basis,
the work-vector arena and the GMRES scalar block are all allocated by the
Python side and passed in as addresses; the only allocation inside Mojo is the
linked-list scratch the symbolic phase needs, which mirrors PETSc's own
`PetscMalloc` calls in the same functions. The GMRES scalar block is one
allocation with fixed offsets, documented at the top of `src/ksp.mojo`:

```
HH        (max_k + 2) * (max_k + 1), column major
HES       (max_k + 1) * (max_k + 1), column major
CC, SS    max_k + 1 each
RS        max_k + 2
ORTHOG    max_k + 2
```

and `HH(a, b) = hh + b * (max_k + 2) + a`, exactly as upstream's macro.

## Tests

`pixi run test` runs three suites plus the conftest path setup:

- `tests/test_vec_parity.py` — every covered `Vec` method against real
  petsc4py, entry for entry, including the cases that distinguish the
  special-cased branches (`axpby` with `beta` 0, `alpha` 0 and `beta` 1;
  `aypx` with `alpha` 0, 1 and -1; the reciprocal zero guard; the infinity-norm
  NaN path).
- `tests/test_mat_parity.py` — the sequential matrix algebra against real
  petsc4py, including the `rows ⊗ cols` semantics of `setValues`.
- `tests/test_pc_parity.py` — `pc.apply(x, y)` for every covered
  preconditioner and level of fill, against real petsc4py running the same
  factorization on the same matrix.
- `tests/test_ksp_parity.py` — the solvers against real petsc4py, compared on
  the iterate, the iteration count, the converged reason, the reported residual
  norm and the residual history, with and without preconditioning, on both
  sides.

## License

MIT, as upstream. See `LICENSE`.

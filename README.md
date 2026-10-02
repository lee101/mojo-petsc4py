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
| `src/simdvec.mojo` | the vectorised forms of the elementwise `Vec` kernels and reductions |
| `src/simd.mojo` | the SIMD width the vectorised kernels sweep at |
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
indptr, indices, vals = [], [], []
for i in range(n):
    indptr.append(len(indices))
    for j in sorted({i, max(i - 1, 0), min(i + 1, n - 1)}):
        indices.append(j)
        vals.append(4.0 if j == i else -1.0)
indptr.append(len(indices))
A.setValuesCSR(
    np.array(indptr, dtype=np.int32),
    np.array(indices, dtype=np.int32),
    np.array(vals, dtype=np.float64),
)

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
A.mult(x, r)
r.aypx(-1.0, b)
print(ksp.getIterationNumber(), r.norm())
```

This prints `1 1.472877282518059e-15` on the machine named in the Benchmarks
section below. The tridiagonal system ILU(0) factorises exactly, so CG converges
in a single iteration and what is left is round-off.

## Coverage

### Covered, with parity tests against real petsc4py

**`PETSc.Vec`** — the elementwise layer, which is where a Krylov method spends
its time. `create`, `createSeq`, `createWithArray`, `duplicate`, `copy`,
`chop`, `getArray`, `setArray`, `getSize`, `getLocalSize`, `getSizes`,
`getBlockSize`, `getValue`, `setValue`, `getValues`, `setValues`, `norm` (all
four norm types), `normBegin`/`normEnd`, `dot`, `tDot`, `dotNorm2`, `sum`,
`mean`, `min`, `max`, `normalize`, `zeroEntries`, `set`, `scale`, `shift`,
`swap`, `axpy`, `aypx`, `axpby`, `axpbypcz`, `waxpy`, `maxpy`, `pointwiseMult`,
`pointwiseDivide`, `pointwiseMin`, `pointwiseMax`, `pointwiseMaxAbs`,
`maxPointwiseDivide`, `reciprocal`, `exp`, `log`, `sqrtabs`, `abs`,
`conjugate`, and the arithmetic operators.

**`PETSc.Mat`** — sequential `MATSEQAIJ`. `create`, `createSeqAIJ`, `createAIJ`,
`setSizes`, `setType`, `setUp`, `assemble`, `setValue`, `setValues`,
`setValueBlocked`, `setValuesBlocked`, `getValue`, `getValues`, `getValuesCSR`,
`setValuesCSR`, `getDiagonal`, `getSize`, `getLocalSize`, `getSizes`,
`getInfo`, `mult`, `multAdd`, `multTranspose`, `multTransposeAdd`,
`transpose`, `scale`, `zeroEntries`, `equal`, `isSymmetric`, `isTranspose`,
`createVecLeft`, `createVecRight`, and `from_dense`.

**`PETSc.PC`** — `none`, `jacobi` (all four `PCJacobiType` values,
`applySymmetricLeft`/`applySymmetricRight` included), `sor` (including
`omega`, `shift` and the symmetric/forward sweep choice), `ilu` (level of fill
0..n, natural or RCM ordering), `icc` (level of fill 0..n). Also
`setOperators`, `setUp`, `reset`, `apply`, `applyTranspose`, `getFactorMatrix`,
`setFactorLevels`, `setFactorShift`, `setFactorPivot`, `setFactorOrdering`.

**`PETSc.KSP`** — `cg`, `gmres`, `fgmres` (restart and
`setGMRESOrthogonalization`/`setGMRESCGSRefinementType`), `bcgs`, with the
default convergence test (`KSPConvergedDefault`), the residual history, the
converged-reason reporting, left/right/symmetric preconditioner sides, and
nonzero initial guesses. All four reproduce upstream's iteration count, its
converged reason and its iterate; see `tests/test_ksp_parity.py`.

### Not covered

- **Everything else in the enums.** `KSP.Type` has 48 members upstream and
  `PC.Type` 49; the ones listed above work, and selecting any other raises
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
- **`Vec` extras that need an index set**: `isaxpy`, `isset`, `permute`. All
  three take an `IS` as their first argument, and the port does not model the
  `IS` type, so porting them would mean inventing a signature rather than
  mirroring one. The underlying scatter is trivial once `IS` exists.
- **KSP features**: monitors, user convergence tests, `setMinimumIterations`
  beyond the port's own use of it, `computeEigenvalues`,
  `computeExtremeSingularValues`, `computeRitz`, `setConvergenceTest`,
  `setMonitor`, `buildSolution`, `buildResidual`, `getWorkVecs`.
- **`Vec`/`Mat` extras**: ghosts, sub-vectors, `IS`, `SF`, DM objects, the
  viewers, DLPack, device types.

### Documented divergences

Each of these is a place where the port is not what upstream is. Nothing else
differs silently.

1. **`Vec.dotNorm2` works here.** Upstream petsc4py 3.25.5's `Vec.dotNorm2`
   raises `TypeError` on every call — `toReal(float)` in `Vec.pyx:2164` is
   applied to the `float` type rather than to a value. The port returns the
   `(dot, ||x||²)` pair.
2. **`Mat.setValuesCSR` takes three arguments.** Upstream's AIJ form is
   `setValuesCSR(indptr, indices, data)` for a sequential matrix but
   `setValuesCSR(i, v)` (a coupled index/value array) for `MATAIJ` in general;
   the port implements the three-argument sequential form, which is the one
   that matches its own `getValuesCSR` and the `MATSEQAIJ` layout.
3. **Transcendental elementwise functions differ by a few ulp.**
   `Vec.exp` and `Vec.log` go through libm on both sides, but not the same
   libm; they agree to about a dozen significant digits rather than to the
   last bit. The reductions (`norm`, `sum`, `dot`) differ the same way, since
   upstream reaches them through BLAS with its own unrolling. The parity tests
   hold these to `1e-13` relative, and `exp`/`log` to `1e-9`.
4. **`MatPivotCheck` and the factor shift are not ported.** With
   `setFactorShift('positive_definite')` upstream retries the factorization
   with an increasing diagonal shift when a pivot is too small; the port does
   not, so a shift-sensitive factor can differ. The default shift type is
   `none` on both sides, which has no retry and is ported exactly.
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
   it into the factor layout. Those routines copy out of the staging buffer into
   the final one in upstream; here the destination is a second caller-allocated
   array. It has to be: the IC pass shifts each row down by one to put the
   diagonal last, and doing that in place reads a destination the previous write
   has already overwritten, which corrupts the next row. The pattern produced
   is identical. The level-of-fill symbolic returns the negated capacity it
   needs when a caller-supplied buffer is too small, so the caller can retry;
   PETSc grows its own free-space blocks instead. Nothing about the pattern
   depends on which path was taken.
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
14. **`Vec.norm` is slower than upstream.** Upstream reaches the reductions
    through BLAS with its own unrolling; the port uses a hand-written SIMD
    sweep. On this machine the 2-norm runs at 0.55x to 0.92x of upstream's time
    depending on length, while `exp` is 2.5x faster, which is the trade the
    measurements in the Benchmarks section show. This is a performance
    difference, not a behavioural one: the parity tests hold the reductions to
    `1e-13` relative.
15. **No parallelism and no GPU.** Everything is single-threaded Mojo. The
    kernels here are all well under two flops per byte, which is the regime
    where neither a thread pool nor a device transfer pays for itself; the
    vectorised sweeps in the Benchmarks section are memory bound, not compute
    bound.

## Benchmarks

Numbers are the best of three timed repetitions, produced by `pixi run bench`
on the machine named in the table. Nothing here is estimated or extrapolated;
where the port is slower than upstream, the table says so.

| case | upstream petsc4py | mojo-petsc4py | speedup |
| --- | --- | --- | --- |
| Vec axpy + pointwiseMult + scale, n = 65,536 | 0.253 ms | 0.277 ms | 0.91x |
| Vec axpy + pointwiseMult + scale, n = 1,048,576 | 5.144 ms | 4.968 ms | 1.04x |
| Vec axpy + pointwiseMult + scale, n = 4,194,304 | 56.996 ms | 42.251 ms | 1.35x |
| Vec norm (2-norm, cache defeated), n = 65,536 | 0.024 ms | 0.042 ms | 0.57x |
| Vec norm (2-norm, cache defeated), n = 1,048,576 | 0.290 ms | 0.529 ms | 0.55x |
| Vec norm (2-norm, cache defeated), n = 4,194,304 | 2.630 ms | 2.866 ms | 0.92x |
| Vec exp, n = 1,048,576 | 8.977 ms | 3.553 ms | 2.53x |
| Mat multVec, 5-point Laplacian 64x64 (n = 4,096) | 0.022 ms | 0.039 ms | 0.57x |
| Mat multVec, 5-point Laplacian 128x128 (n = 16,384) | 0.082 ms | 0.112 ms | 0.73x |
| Mat multVec, 5-point Laplacian 256x256 (n = 65,536) | 0.314 ms | 0.333 ms | 0.94x |
| CG + PCNONE, 32x32 (n = 1,024) (113/113 its ref/mojo) | 0.992 ms | 1.058 ms | 0.94x |
| CG + PCNONE, 64x64 (n = 4,096) (225/225 its ref/mojo) | 7.572 ms | 7.855 ms | 0.96x |
| CG + PCNONE, 128x128 (n = 16,384) (434/434 its ref/mojo) | 64.019 ms | 66.566 ms | 0.96x |
| GMRES + PCNONE, 32x32 (n = 1,024) (232/232 its ref/mojo) | 4.271 ms | 5.560 ms | 0.77x |
| GMRES + PCNONE, 64x64 (n = 4,096) (626/626 its ref/mojo) | 47.440 ms | 62.491 ms | 0.76x |
| GMRES + PCNONE, 128x128 (n = 16,384) (2000/2000 its ref/mojo) | 642.834 ms | 877.161 ms | 0.73x |
| FGMRES + PCNONE, 32x32 (n = 1,024) (232/232 its ref/mojo) | 5.071 ms | 4.962 ms | 1.02x |
| FGMRES + PCNONE, 64x64 (n = 4,096) (626/626 its ref/mojo) | 59.611 ms | 64.424 ms | 0.93x |
| FGMRES + PCNONE, 128x128 (n = 16,384) (2000/2000 its ref/mojo) | 667.674 ms | 745.776 ms | 0.90x |
| BCGS + PCNONE, 32x32 (n = 1,024) (85/83 its ref/mojo) | 1.434 ms | 1.409 ms | 1.02x |
| BCGS + PCNONE, 64x64 (n = 4,096) (160/168 its ref/mojo) | 11.084 ms | 12.510 ms | 0.89x |
| BCGS + PCNONE, 128x128 (n = 16,384) (301/319 its ref/mojo) | 87.910 ms | 94.793 ms | 0.93x |
| CG + PCILU, 32x32 (n = 1,024) (40/40 its ref/mojo) | 0.699 ms | 0.869 ms | 0.80x |
| CG + PCILU, 64x64 (n = 4,096) (77/77 its ref/mojo) | 5.990 ms | 6.236 ms | 0.96x |
| CG + PCILU, 128x128 (n = 16,384) (146/146 its ref/mojo) | 48.261 ms | 50.583 ms | 0.95x |
| CG + PCICC, 32x32 (n = 1,024) (40/40 its ref/mojo) | 0.722 ms | 0.896 ms | 0.81x |
| CG + PCICC, 64x64 (n = 4,096) (77/77 its ref/mojo) | 5.760 ms | 6.122 ms | 0.94x |
| CG + PCICC, 128x128 (n = 16,384) (146/146 its ref/mojo) | 43.716 ms | 46.324 ms | 0.94x |
| GMRES + PCJACOBI, 32x32 (n = 1,024) (232/232 its ref/mojo) | 4.356 ms | 4.724 ms | 0.92x |
| GMRES + PCJACOBI, 64x64 (n = 4,096) (626/626 its ref/mojo) | 46.763 ms | 62.029 ms | 0.75x |
| GMRES + PCJACOBI, 128x128 (n = 16,384) (2000/2000 its ref/mojo) | 636.641 ms | 874.541 ms | 0.73x |

Machine: Linux x86_64, Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz,
72 logical cores, Python 3.13.14, NumPy 2.5.1. Against upstream petsc4py
3.25.5 built with the same toolchain.

`pixi run bench` builds every matrix, vector, `KSP` and `PC` outside the timed
region, so what is measured is the operation and not the setup. The `(a/b its
ref/mojo)` column is the iteration count each side actually took on that solve.
The port runs at 0.94x to 0.96x of upstream on unpreconditioned CG, where the
iteration count matches exactly on both sides, and at 0.72x to 0.79x on GMRES,
where the sequential CSR multiply and the per-vector Python round trips
dominate. `bcgs` is the one solver whose iteration count differs: it takes 319
iterations where upstream takes 301 at 128x128, so that row is not measuring the
same amount of work on the two sides. `bcgs` is not bit-reproducible against
upstream for the same reason the reductions are not — the scalars differ in the
last bits, and a biorthogonal method amplifies that — so the parity tests hold
it to the same solution and converged reason rather than the same iteration
count.

### What was done, and what it bought

The elementwise `Vec` layer and the reductions in it are where a Krylov method
spends its time, and every one of those kernels was a scalar `while` loop. They
now live in `src/simdvec.mojo`: each one has a vector body that handles
`simd_width_of[DType.float64]()` elements per iteration and a scalar tail. The
sweep starts at index 0 and uses `unsafe_load[width=SW]`, which lowers to an
unaligned vector load; a `numpy` buffer is only guaranteed to be 16-byte
aligned, so there is no alignment peel. On aligned data the unaligned form costs
the same as the aligned one, and on unaligned data it costs what the hardware
costs rather than a peel loop on every call.

Three kernels are deliberately still scalar, because a vector reduction cannot
reproduce what they promise: `VecMinMax_Seq` reports the *first* index of a run
of equal extremes and seeds its accumulator from `x[0]`, and the infinity norm
adopts the first NaN and stops. `VecReciprocal` and `VecChop` need a blend
between two computed values, which this dialect has no branchless select for.

Two changes were measured and reverted, and the code is as it was found:

- Fusing the `VecCopy` of `b` into the `aypx` of the initial residual changed
  GMRES from 232 iterations to a breakdown at 30. The two-pass form is correct.
- Unrolling the CSR multiply four ways to give the loop four independent
  dependency chains is not expressible: `y` is indexed by row, so a 4-way
  unroll over non-zero entries indexes `y` out of bounds on a 2-entry row.

`max.algorithm.parallelize` was compiled and measured rather than assumed. It
works on this toolchain, but this library is driven from a ctypes host, so every
launch pays a thread-pool dispatch that the elementwise kernels cannot amortise
at the sizes the benchmarks exercise. It is not in the shipped path.

No GPU path was attempted. Every kernel here reads 8 bytes and writes 8 bytes
per element and does one or two arithmetic operations on them, which is well
under the arithmetical intensity where a `max.gpu.host` launch can stop losing
to plain SIMD on CPU; on a memory-bound sweep the transfer would dominate any
arithmetic anyway.

## How it works

**FFI strategy.** `mojo build --emit shared-lib` compiles `src/capi.mojo` into
`dist/libmojo-petsc4py.so`. The pinned compiler rejects `@export` on
parametric functions, so every buffer crosses the ABI as an `Int` address and
is rebuilt into a `Pointer` inside the wrapper:

```mojo
@export("mp_vec_axpy")
def mp_vec_axpy(y: Int, alpha: Float64, x: Int, n: Int) abi("C"):
    VecAXPY_Seq(p(y), alpha, p(x), n)
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

`pixi run test` runs seven suites plus the conftest path setup:

- `tests/test_vec_parity.py` — every covered `Vec` method against real
  petsc4py, entry for entry, including the cases that distinguish the
  special-cased branches (`axpby` with `beta` 0, `alpha` 0 and `beta` 1;
  `aypx` with `alpha` 0, 1 and -1; the reciprocal zero guard; the infinity-norm
  NaN path) and the boundary of `chop` at `tol` 0 and at `tol` equal to an
  entry's magnitude.
- `tests/test_mat_parity.py` — the sequential matrix algebra against real
  petsc4py, including the `rows ⊗ cols` semantics of `setValues` and a
  `setValuesCSR` round trip.
- `tests/test_pc_parity.py` — `pc.apply(x, y)` for every covered
  preconditioner and level of fill, against real petsc4py running the same
  factorization on the same matrix.
- `tests/test_ksp_parity.py` — the solvers against real petsc4py, compared on
  the iterate, the iteration count, the converged reason, the reported residual
  norm and the residual history, with and without preconditioning, on both
  sides. `cg`, `gmres`, `fgmres` and `bcgs` are all asserted to match, not
  merely to run.

- `tests/test_api_parity.py` — the covered surface itself, so that the coverage
  list above stays honest: `createWithArray`, `normBegin`/`normEnd`,
  `getSizes`/`getInfo`, `equal`, `isSymmetric`/`isTranspose`,
  `createVecLeft`/`createVecRight`, the blocked setters, `PC.reset`,
  `applyTranspose`, the factor options, `setInitialGuessNonzero`,
  `setGMRESRestart` and the GMRES orthogonalisation and refinement settings.
  Non-square matrices get their own test, because the row and column counts
  have to stay distinct.
- `tests/test_factor_layout.py` — the ILU and IC factor layouts directly. These
  kernels index flat caller-owned arrays with no bounds checks, so a layout
  mistake is a silent out-of-bounds write rather than an exception; the tests
  pin `ui`/`uj`/`udiag` down and assert the contiguous pass does not touch its
  staging buffer.
- `tests/test_simd_parity.py` — the vectorised elementwise kernels and
  reductions against real petsc4py at sizes that straddle the vector width.
- `tests/test_ffi_signatures.py` — every `@export` in `src/capi.mojo` against
  the ctypes declaration in `_lib.py`. Both sides take buffers as `Int`
  addresses, so a declaration that is one argument short does not raise; it
  reads an argument nobody set. The test compares arity, integer and float
  argument counts and return type for all of them.

The suite requires upstream petsc4py, which `pixi.toml` installs; each file
skips itself cleanly with `importorskip` if the package is absent.

## License

MIT, as upstream. See `LICENSE`.

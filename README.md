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

This prints `1 1.4432899320127035e-15` on the machine named in the Benchmarks
section below (reproduced exactly on three consecutive runs). The tridiagonal
system ILU(0) factorises exactly, so CG converges in a single iteration and
what is left is round-off.

`_lib.py` loads `dist/libmojo-petsc4py.so` and rebuilds it with `mojo` when
`src/` is newer, so `pixi run build` before the first import is a convenience
rather than a requirement. `pyproject.toml` packages the Python tree only: a
plain `pip install` of this repository would import the package but would not
compile the library, so the supported path is the `pixi` environment above
(with `mojo` on `PATH`, or `MOJOPETSC4PY_MOJO` pointing at it).

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
nonzero initial guesses. `cg`, `gmres` and `fgmres` reproduce upstream's
iteration count, its converged reason and its iterate when they are driven the
way `bench.py` and the tests drive them -- a `PC` set on the solver and then
`setOperators`. `bcgs` reproduces the converged reason and the solution but can
take a few more iterations (see divergence 16). One portability limit: a `KSP`
used without ever calling `setPC` is not equivalent to upstream's, because
upstream then has no preconditioner object at all; attach a `PC` (even
`none`), as the example above does.

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
7. **The level-of-fill symbolic phase needs its caller to size two buffers.**
   `MatILUFactorSymbolic_SeqAIJ` / `MatICCFactorSymbolic_SeqAIJ` return the
   negated size they need when the staging `cap` passed in is too small, and
   the IC variant also needs `2 * cap` entries of merge workspace after its
   `6 * (n + 1)` fixed slots, reported the same way; the caller allocates and
   retries. PETSc grows its own free-space blocks instead. The pattern does not
   depend on which path was taken.

   The IC contiguous pass (`PetscFreeSpaceContiguous_Cholesky`) copies into
   `b->j` in place, which is safe because each row is shifted down by one
   within itself; the ILU one (`PetscFreeSpaceContiguous_LU`) needs distinct
   source and destination buffers, exactly as upstream does.
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
14. **The reductions sum in a different order from upstream's BLAS.** The
    four-accumulator sweep groups elements into blocks of 16; upstream's
    `BLASdot`/`BLASnrm2` unroll by their own amount. Both are plain sums, and
    they agree to the last few bits (at most `1e-14` relative on this machine),
    which the parity tests hold to `1e-13`. The 2-norm was the port's slowest
    operation relative to upstream before the four-accumulator reductions and
    the cached addresses; in the published table it is 0.71x to 1.37x, which the
    run-to-run spread above covers.
15. **No parallelism and no GPU.** The library is single-threaded. The
    experiments behind this used `max.algorithm.parallelize` from a ctypes host
    and measured a launch cost of 5 to 7.5 ms, against under 2 ms of the
    multiply itself at every size the benchmarks reach, so a parallel variant
    was slower and is not shipped. That API is not importable in the locked
    environment (`max.algorithm` is absent), so the figure is a record of the
    earlier measurement rather than something this tree can re-run. The
    kernels are all well under two flops per byte, the regime where a device
    transfer does not pay either; no GPU path was built.
16. **BCGS can take a different number of iterations from upstream.** The
    recurrence is transcribed from `KSPSolve_BCGS`, and on the tests' small SPD
    matrices the count matches; on the 5-point Laplacian with no preconditioner
    it can differ by a few iterations in either direction (160/162 at 64x64 and
    301/316 at 128x128 in the table above), while the converged reason, the
    tolerance and the solution agree. The count is decided by which of two
    nearly equal residual estimates crosses `rtol` first, so this is the
    reduction-order divergence (14) showing up in the solver rather than a
    different algorithm. `tests/test_ksp_parity.py` pins that shape.

## Benchmarks

Numbers are the best of three timed repetitions, produced by `pixi run bench`
on the machine named here. Nothing is estimated or extrapolated; where the port
is slower than upstream, the table says so.

Measured on Linux x86_64, Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 logical cores,
Python 3.13.14, NumPy 2.5.1, upstream PETSc 3.25.5, with a load average near 32.
This is the full output of `pixi run bench` on the current tree. The machine is
shared, so treat the ratios as indicative, not exact: re-running the 4M `Vec`
rows over 15 repetitions gave a 1.4-3.0 ms range for `norm` on both sides
(median 2.6 ms upstream, 2.2 ms here), which is wider than several of the
differences in the table. `bench/direct.py` runs the same cases with nine
repetitions and reports the median, and `bench/compare_states.py` alternates two
builds for a before/after comparison.

| case | upstream petsc4py | mojo-petsc4py | speedup |
| --- | --- | --- | --- |
| Vec axpy + pointwiseMult + scale, n = 65,536 | 0.192 ms | 0.243 ms | 0.79x |
| Vec axpy + pointwiseMult + scale, n = 1,048,576 | 4.576 ms | 3.547 ms | 1.29x |
| Vec axpy + pointwiseMult + scale, n = 4,194,304 | 48.164 ms | 42.694 ms | 1.13x |
| Vec norm (2-norm, cache defeated), n = 65,536 | 0.021 ms | 0.031 ms | 0.69x |
| Vec norm (2-norm, cache defeated), n = 1,048,576 | 0.344 ms | 0.274 ms | 1.26x |
| Vec norm (2-norm, cache defeated), n = 4,194,304 | 2.747 ms | 2.265 ms | 1.21x |
| Vec exp, n = 1,048,576 | 8.419 ms | 3.226 ms | 2.61x |
| Mat multVec, 5-point Laplacian 64x64 (n = 4,096) | 0.022 ms | 0.036 ms | 0.60x |
| Mat multVec, 5-point Laplacian 128x128 (n = 16,384) | 0.082 ms | 0.106 ms | 0.78x |
| Mat multVec, 5-point Laplacian 256x256 (n = 65,536) | 0.336 ms | 0.343 ms | 0.98x |
| CG + PCNONE, 32x32 (n = 1,024) (113/113 its ref/mojo) | 0.885 ms | 0.895 ms | 0.99x |
| CG + PCNONE, 64x64 (n = 4,096) (225/225 its ref/mojo) | 7.814 ms | 7.597 ms | 1.03x |
| CG + PCNONE, 128x128 (n = 16,384) (434/434 its ref/mojo) | 62.096 ms | 65.590 ms | 0.95x |
| GMRES + PCNONE, 32x32 (n = 1,024) (232/232 its ref/mojo) | 4.281 ms | 3.580 ms | 1.20x |
| GMRES + PCNONE, 64x64 (n = 4,096) (626/626 its ref/mojo) | 48.915 ms | 57.955 ms | 0.84x |
| GMRES + PCNONE, 128x128 (n = 16,384) (2000/2000 its ref/mojo) | 627.851 ms | 720.996 ms | 0.87x |
| FGMRES + PCNONE, 32x32 (n = 1,024) (232/232 its ref/mojo) | 3.863 ms | 3.233 ms | 1.19x |
| FGMRES + PCNONE, 64x64 (n = 4,096) (626/626 its ref/mojo) | 44.095 ms | 47.752 ms | 0.92x |
| FGMRES + PCNONE, 128x128 (n = 16,384) (2000/2000 its ref/mojo) | 643.868 ms | 693.121 ms | 0.93x |
| BCGS + PCNONE, 32x32 (n = 1,024) (85/85 its ref/mojo) | 1.475 ms | 1.249 ms | 1.18x |
| BCGS + PCNONE, 64x64 (n = 4,096) (160/162 its ref/mojo) | 11.617 ms | 10.906 ms | 1.07x |
| BCGS + PCNONE, 128x128 (n = 16,384) (301/316 its ref/mojo) | 88.869 ms | 91.673 ms | 0.97x |
| CG + PCILU, 32x32 (n = 1,024) (40/40 its ref/mojo) | 0.797 ms | 0.784 ms | 1.02x |
| CG + PCILU, 64x64 (n = 4,096) (77/77 its ref/mojo) | 6.092 ms | 6.067 ms | 1.00x |
| CG + PCILU, 128x128 (n = 16,384) (146/146 its ref/mojo) | 47.202 ms | 48.259 ms | 0.98x |
| CG + PCICC, 32x32 (n = 1,024) (40/40 its ref/mojo) | 0.694 ms | 0.828 ms | 0.84x |
| CG + PCICC, 64x64 (n = 4,096) (77/77 its ref/mojo) | 5.638 ms | 6.008 ms | 0.94x |
| CG + PCICC, 128x128 (n = 16,384) (146/146 its ref/mojo) | 43.424 ms | 48.678 ms | 0.89x |
| GMRES + PCJACOBI, 32x32 (n = 1,024) (232/232 its ref/mojo) | 4.275 ms | 4.485 ms | 0.95x |
| GMRES + PCJACOBI, 64x64 (n = 4,096) (626/626 its ref/mojo) | 46.950 ms | 51.261 ms | 0.92x |

### What was done, and what it bought

The elementwise `Vec` layer and the reductions in it are where a Krylov method
spends its time. Four changes were kept, each measured against the build
before it; two were measured and dropped.

**The per-call cost was the first thing.** Every `Vec` method resolved each
buffer's address with `np.ndarray.ctypes.data`, which rebuilds the array's
ctypes descriptor on each access (about 1.8 us here) against roughly 1.3 us for
the FFI call it feeds, and a GMRES iteration resolves a dozen of them. `Vec`,
the `Mat` descriptor and the KSP/PC blocks now keep the address beside the
buffer, and the reduction methods that return a plain float reuse a per-vector
result slot instead of allocating one per call. `bench/compare_states.py` is
the tool for measuring a before/after pair of source states; the specific
ratios it produced at the time of that change are not reproduced here, because
they need the earlier build, which the published table does not measure. The
published table above is `pixi run bench` on the current tree.

**The reductions now carry four accumulators.** A single SIMD accumulator is
one FMA dependency chain: it can issue one load per three cycles, about 25 GB/s
on this core, while the array can be read at 33 GB/s. `simdvec.dot`,
`dot_norm2`, `norm2`, `asum` and `total` accumulate a `BLK`-element block into
four registers, so four loads are in flight. Measured in isolation on this
machine by timing `simdvec.norm2` against a single-accumulator body of the same
loop compiled into the same binary, the four-accumulator loop is 1.9x the
one-accumulator loop at n = 2^16 and n = 2^20, where the array is in cache,
and 1.4x at n = 2^22, where memory bandwidth rather than the FMA chain is the
limit. Operator precedence is unchanged -- the same products,
summed in a different order -- so this moves the last bits of a reduction, and
the parity tolerance on the reductions stays what it was (`1e-13` relative,
which they meet with room; the difference measured at most `1e-14`). The block
is one vector group wide because a wider one is not correct here: a body that
stashed four accumulators over four block offsets built to a single `vaddpd`
and answered a quarter of the truth in a standalone binary, whatever form the
accumulators were declared in, so the width kept is the one that was checked in
the disassembly and against a numpy reference at every length from 0 to 16385.

**Four guarded kernels stayed scalar on purpose.** `VecMinMax_Seq` reports the
*first* index of a run of equal extremes, and the infinity norm adopts the first
NaN and stops; neither survives a reordering. `VecReciprocal` and `VecChop`
need a per-lane select, which this dialect does not have (no `select`, no
`reduce_or` on a boolean SIMD, no lane indexing of one). A group-wide "all
lanes are ordinary" test was written, measured and reverted: it is wrong for a
subnormal, which must reciprocate to infinity, and wrong for an entry exactly
at `chop`'s tolerance, which must be rewritten rather than left alone.

**The GMRES inner products are batched.** Classical Gram-Schmidt's `VecMDot`
computed one dot per basis vector and re-read the new Krylov vector from memory
each time. It now folds four rows at a time through `_mdot_basis`: one load of
each `x` group serves four rows. A row keeps its own block-then-tail sequence,
so its value is the one the single-vector `dot` produced. The gain is small and
does not reproduce at every size: against the unbatched build it came out 0.90x
on n = 16384 (2^14 rows per dot, 2000 iterations) and 1.42x on n = 4096 on one
pair of runs, and 1.009x and 1.036x on another. Above 2^14 the vector is out of
cache and reading it once per four rows is a real saving; at 2^12 it fits in
cache and the extra live accumulators cost about what the saved loads were
worth. The honest statement is that it is neutral to 40% faster depending on
size, not uniformly faster.

**IC's triangular solve drops two passes.** It copied `b` into `x`, ran the
forward sweep in place, and then made two more length-n copies to apply the
column permutation. Upstream's `MatSolve_SeqSBAIJ_1_inplace` gathers `b` into
the work vector as it permutes it and writes `x` directly from the back
substitution, because a row reads only columns that the rows below it have
already written. The port now does the same, three passes instead of five.

Measured and dropped:

- **`max.algorithm.parallelize` for the CSR multiply.** In the earlier
  measurement it compiled and ran from a ctypes host on this toolchain, at a
  launch cost of 5.0 to 7.5 ms here, flat, against 0.1 to 1.9 ms of matvec work
  at n = 16384 to 262144, so every parallel variant was 5x to 20x *slower* than
  the serial loop. (The API is not importable in the locked environment, so
  this is not re-runnable here.) It only pays
  once a kernel runs for tens of milliseconds, which no covered operation does
  at the sizes callers pass. The same launch cost makes it lose on `Vec` too.
- **Fusing the `VecCopy` of `b` into the `aypx` of the initial residual**, from
  the earlier work: it changed GMRES from 232 iterations to a breakdown at 30.
  The two-pass form is correct.

A GPU path was not built. Every covered kernel moves 16 bytes per element and
does one or two operations on them, far below the intensity at which a
`max.gpu.host` launch can beat SIMD on a CPU, and on a memory-bound sweep the
transfer would dominate the arithmetic. (`max.algorithm` and `max.gpu` are not
importable in the locked environment; this is an argument from the kernels'
arithmetic intensity, not a measurement of a device path, since no device path
was built.)

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

**Factor layouts.** The two incomplete factorisations use PETSc's own layouts,
and a reader checking the port against `aijfact.c` needs them stated, because
they are not the obvious ones:

- ILU. `bj` holds every L row in increasing order, then every U row in
  decreasing order, each U row ending on its diagonal, which is stored
  inverted. `bi` delimits the L block and `bdiag[i]` is that row's diagonal,
  counted back from the end of the array. This is the layout
  `MatILUFactorSymbolic_SeqAIJ`'s header comment documents, and
  `MatLUFactorNumeric_SeqAIJ` / `MatSolve_SeqAIJ` are transcribed against it.
- IC. The factor is `MATSEQSBAIJ`, storing the strictly upper triangle of `A`
  with each row's diagonal slot last, holding `1/d_i`. `udiag[i] ==
  b->i[i + 1] - 1`. This is what `MatGetRowIJ` reports on a real petsc4py IC
  factor (checked directly: for a tridiagonal matrix it prints
  `ja = [1, 0, 2, 1, 3, 2, ...]`), and `MatCholeskyFactorNumeric_SeqAIJ` /
  `MatSolve_SeqSBAIJ_1_NaturalOrdering` are transcribed against it.

## Tests

`pixi run test` runs eleven suites (594 tests) plus the conftest path setup:

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
  mistake is a silent out-of-bounds write rather than an exception. The tests
  pin down the IC factor's row shape (strictly upper columns ascending, then
  the diagonal slot), the `b->i`/`b->diag` relationship the numeric pass and
  the triangular solve both read, and that the contiguous pass leaves the
  staging map `ui` alone.
- `tests/test_simd_parity.py` — the vectorised elementwise kernels and
  reductions against real petsc4py at sizes that straddle the vector width.
- `tests/test_block_reductions.py` — the reductions at the lengths that enter
  their four-accumulator block (0, one, `SW`-1, `SW`, `SW`+1 and every boundary
  up to 16385) against both NumPy over the whole array and upstream, so a block
  that is skipped or double-counted fails on the value; the guarded reciprocal
  at a zero, a subnormal and a NaN; `chop` at exactly its tolerance; and a
  GMRES/FGMRES solve through the batched inner products, checked on the iterate
  and the iteration count.
- `tests/test_addr_cache.py` — the cached FFI addresses. Every public path that
  rebinds a vector's storage is walked and the reported address must be the new
  buffer's, because a stale one is a dangling pointer rather than an exception.
- `tests/test_exported_kernels.py` — every `@export`ed kernel called by name,
  with the argument blocks built from the documented layouts and its output
  checked against an independent NumPy reference or against real petsc4py.
  The parity suites reach these kernels through the public API, which cannot
  notice a kernel that is wired up but wrong; this file closes that hole, and
  the check is mechanical: each `@export("name")` in `src/capi.mojo` must
  appear as a literal in the file. It also asserts that the library that was
  loaded really is `dist/libmojo-petsc4py.so`, so no test can pass on a
  pure-Python fallback.
- `tests/test_ffi_signatures.py` — every `@export` in `src/capi.mojo` against
  the ctypes declaration in `_lib.py`. Both sides take buffers as `Int`
  addresses, so a declaration that is one argument short does not raise; it
  reads an argument nobody set. The test compares arity, integer and float
  argument counts and return type for all of them.

The suite requires upstream petsc4py, which `pixi.toml` installs; each file
skips itself cleanly with `importorskip` if the package is absent.

## License and provenance

The original code in this repository -- the Python packages, the Mojo modules,
the build and benchmark scripts and the tests -- is MIT; see `LICENSE`. It is a
transcription, not a fork: the algorithm in each ported function is derived from
PETSc / petsc4py source, and that derivation is recorded here and in `NOTICE`.
The upstream material is 2-clause BSD, which is compatible with keeping the
original port under MIT; the upstream copyright notices and licence conditions
are retained in `NOTICE` and in the two blocks below, because the BSD licence
requires them to accompany redistributions.

| upstream | URL and revision | license | derived into |
| --- | --- | --- | --- |
| petsc4py `PETSc/{Vec,Mat,PC,KSP}.pyx` and the `PETSc` Python API it defines | <https://gitlab.com/petsc/petsc/-/tree/v3.25.5/src/binding/petsc4py> tag `v3.25.5` | BSD-2-Clause (`LICENSE.rst` in that directory) | `python/mojopetsc4py/PETSc/`, `src/ported.mojo`, `src/ksp.mojo` |
| PETSc `src/mat/impls/aij/seq/{aij.c,aijfact.c}`, `src/mat/impls/sbaij/seq/sbaijfact2.c`, `src/mat/utils/freespace.c`, `include/petsc/private/matimpl.h` | <https://gitlab.com/petsc/petsc> tag `v3.25.5` | BSD-2-Clause (`LICENSE` at the repository root) | `src/ported.mojo`, `src/factor.mojo` |
| PETSc `src/ksp/ksp/impls/{cg,gmres,fgmres,bcgs}`, `src/ksp/ksp/interface/itres.c`, `src/ksp/pc/impls/{jacobi,sor}` | <https://gitlab.com/petsc/petsc> tag `v3.25.5` | BSD-2-Clause | `src/ksp.mojo`, `src/ported.mojo` |

The parity tests and benchmarks run against the upstream packages but do not
copy them: `petsc4py` and `petsc` are pip/conda dependencies, not vendored, and
nothing from their source tree is included in this one.

**PETSc** (from `LICENSE` at tag `v3.25.5`; also in `NOTICE`):

```
Copyright (c) 1991-2025, UChicago Argonne, LLC and the PETSc Developers and Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

**petsc4py** (from `src/binding/petsc4py/LICENSE.rst` at tag `v3.25.5`; also in `NOTICE`):

```
Copyright (c) 2025, Lisandro Dalcin.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

No upstream dataset is bundled: every test and benchmark matrix is generated
at run time from its definition (a 1-D or 5-point 2-D Laplacian, a random
symmetric positive-definite matrix from a fixed seed), so there is no data
whose redistribution terms would need checking.

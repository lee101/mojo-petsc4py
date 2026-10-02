"""Port of the compute-bound core of petsc4py's Krylov solvers and preconditioners.

Section order follows `include "..."` order in upstream's
`src/petsc4py/PETSc/PETSc.pyx`: `Vec.pyx`, `Mat.pyx`, `PC.pyx`, `KSP.pyx`.  Within a
section the functions appear in the order they appear upstream.

Each function keeps the name and the argument order of the PETSc C routine that
the corresponding petsc4py binding method calls: `Vec.axpy` -> `VecAXPY_Seq` ->
`vec_axpy` here, taking `(y, alpha, x, n)` exactly as `VecAXPY` does.  `n` is the
`VecLayout` size PETSc keeps inside the object; it is threaded explicitly because
Mojo's `@export` cannot accept a struct.

Scalar is Float64, matching the PETSc real-double build this port is tested
against, and `Int` arrays stand in for `PetscInt`.

Structs are avoided throughout: the pinned compiler requires explicit `move`
arguments and an `ImplicitlyCopyable` conformance that a state struct of this
shape does not get for free.  Solver state therefore travels as a caller-owned
scalar block (`st`, `sit`) exactly as PETSc's `ksp->its` / `ksp->rnorm` do, and
the work vectors travel as one flat `Float64` arena sliced by index.
"""

from std.math import abs, exp, log, sqrt

import simdvec

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime PI = Pointer[Int, AnyOrigin[mut=True]]

# petscvec.h NormType, kept as the C ordinals.
comptime NORM_1: Int = 0
comptime NORM_2: Int = 1
comptime NORM_FROBENIUS: Int = 2
comptime NORM_INFINITY: Int = 3
comptime NORM_1_AND_2: Int = 4

# petscksp.h KSPConvergedReason.
comptime KSP_CONVERGED_ITERATING: Int = 0
comptime KSP_CONVERGED_RTOL: Int = 2
comptime KSP_CONVERGED_ATOL: Int = 3
comptime KSP_CONVERGED_ITS: Int = 4
comptime KSP_CONVERGED_NEG_CURVE: Int = 5
comptime KSP_CONVERGED_STEP_LENGTH: Int = 6
comptime KSP_CONVERGED_HAPPY_BREAKDOWN: Int = 7
comptime KSP_DIVERGED_NULL: Int = -2
comptime KSP_DIVERGED_ITS: Int = -3
comptime KSP_DIVERGED_DTOL: Int = -4
comptime KSP_DIVERGED_BREAKDOWN: Int = -5
comptime KSP_DIVERGED_BREAKDOWN_BICG: Int = -6
comptime KSP_DIVERGED_NONSYMMETRIC: Int = -7
comptime KSP_DIVERGED_INDEFINITE_PC: Int = -8
comptime KSP_DIVERGED_NANORINF: Int = -9
comptime KSP_DIVERGED_INDEFINITE_MAT: Int = -10
comptime KSP_DIVERGED_PC_FAILED: Int = -11

# petscpc.h PCSide.
comptime PC_LEFT: Int = 0
comptime PC_RIGHT: Int = 1
comptime PC_SYMMETRIC: Int = 2

comptime MAX_REAL: Float64 = 1.7976931348623157e308
comptime MIN_REAL: Float64 = -1.7976931348623157e308


# ====================================================================
#  petsc4py/PETSc/Vec.pyx
# ====================================================================


def VecPointwiseMult_Seq(w: Ptr, x: Ptr, y: Ptr, n: Int):
    """w = x .* y.  bvec2.c VecPointwiseMult_Seq, the plain-store branch."""
    simdvec.pointwise_mult(w, x, y, n)


def VecPointwiseDivide_Seq(w: Ptr, x: Ptr, y: Ptr, n: Int):
    """w = x ./ y.  bvec2.c VecPointwiseApply_Seq with the ScalDiv functor."""
    simdvec.pointwise_divide(w, x, y, n)


def VecPointwiseMin_Seq(w: Ptr, x: Ptr, y: Ptr, n: Int):
    """w = min(x, y).  bvec2.c VecPointwiseMin_Seq via MinRealPart."""
    simdvec.pointwise_min(w, x, y, n)


def VecPointwiseMax_Seq(w: Ptr, x: Ptr, y: Ptr, n: Int):
    """w = max(x, y).  bvec2.c VecPointwiseMax_Seq via MaxRealPart."""
    simdvec.pointwise_max(w, x, y, n)


def VecPointwiseMaxAbs_Seq(w: Ptr, x: Ptr, y: Ptr, n: Int):
    """w = max(|x|, |y|).  bvec2.c VecPointwiseMaxAbs_Seq via MaxAbs."""
    simdvec.pointwise_maxabs(w, x, y, n)


def VecNorm2(x: Ptr, n: Int) -> Float64:
    """The NORM_2 branch of VecNorm_Seq.  Factored out so the callers that need
    the value rather than a `PetscReal *` (VecNormalize, GMRES) can share it."""
    if not n:
        return 0.0
    return sqrt(simdvec.norm2(x, n))


def _BLASasum(x: Ptr, n: Int) -> Float64:
    """BLASasum: the sum of |x_i|."""
    return simdvec.asum(x, n)


def VecNorm_Seq(x: Ptr, ntype: Int, z: Ptr, n: Int):
    """bvec2.c VecNorm_Seq, including the NaN-breaking infinity loop."""
    var ztmp0 = 0.0
    var ztmp1 = 0.0
    if n:
        if ntype == NORM_2 or ntype == NORM_FROBENIUS or ntype == NORM_1_AND_2:
            # upstream indexes by `type == NORM_1_AND_2`, so the 2-norm lands in
            # z[0] on its own and in z[1] alongside the 1-norm
            if ntype == NORM_1_AND_2:
                ztmp0 = _BLASasum(x, n)
                ztmp1 = VecNorm2(x, n)
            else:
                ztmp0 = VecNorm2(x, n)
        elif ntype == NORM_INFINITY:
            # upstream adopts a strictly greater value and, on the first NaN,
            # stores it and leaves the whole loop; simdvec.norm_infinity is
            # that loop verbatim
            ztmp0 = simdvec.norm_infinity(x, n)
        elif ntype == NORM_1:
            ztmp0 = _BLASasum(x, n)
    z.unsafe_store(0, ztmp0)
    if ntype == NORM_1_AND_2:
        z.unsafe_store(1, ztmp1)


def VecDot_Seq(x: Ptr, y: Ptr, val: Ptr, n: Int):
    """val = x . y.  BLASdot over the whole local array."""
    val.unsafe_store(0, simdvec.dot(x, y, n))


def VecDotNorm2_Seq(x: Ptr, y: Ptr, sval: Ptr, rval: Ptr, n: Int):
    """The dot product of x with y and the squared norm of x.

    Upstream `VecDotNorm2` is a separate call to `VecDot` followed by a norm
    (`VecNorm_Norm2`); one pass over the array is observationally identical for
    a contiguous, sequential, real vector and is what the wrapper fuses.
    """
    var r = simdvec.dot_norm2(x, y, n)
    sval.unsafe_store(0, r[0])
    rval.unsafe_store(0, r[1])


def VecSum_Seq(x: Ptr, val: Ptr, n: Int):
    val.unsafe_store(0, simdvec.total(x, n))


def VecMean_Seq(x: Ptr, val: Ptr, n: Int):
    val.unsafe_store(0, simdvec.total(x, n) / Float64(n))


def VecMinMax_Seq(x: Ptr, idx: PI, z: Ptr, minmax: Float64, is_max: Int, n: Int):
    """Shared body of VecMin_Seq / VecMax_Seq.  dvec2.c VecMinMax_Seq.

    The first index attaining the extreme wins, because the comparison is
    strict; an empty vector reports index -1 and leaves the sentinel.
    """
    var j = -1
    var mm = minmax
    if n:
        j = 0
        mm = x.unsafe_load(0)
        var i = 1
        while i < n:
            var tmp = x.unsafe_load(i)
            if is_max != 0:
                if tmp > mm:
                    j = i
                    mm = tmp
            else:
                if tmp < mm:
                    j = i
                    mm = tmp
            i += 1
    z.unsafe_store(0, mm)
    idx.unsafe_store(0, j)


def VecMax_Seq(x: Ptr, idx: PI, z: Ptr, n: Int):
    """Seeded with PETSC_MIN_REAL so an all-negative vector still reports one."""
    VecMinMax_Seq(x, idx, z, MIN_REAL, 1, n)


def VecMin_Seq(x: Ptr, idx: PI, z: Ptr, n: Int):
    """Seeded with PETSC_MAX_REAL."""
    VecMinMax_Seq(x, idx, z, MAX_REAL, 0, n)


def VecNormalize_Seq(x: Ptr, v: Ptr, n: Int):
    """x <- x / ||x||₂, returning the norm.  A zero vector is left alone."""
    var z = VecNorm2(x, n)
    if z > 0.0:
        simdvec.normalize_inplace(x, n, z)
    v.unsafe_store(0, z)


def VecReciprocal_Seq(x: Ptr, n: Int):
    """x <- 1/x, with upstream's zero guard: an exact zero becomes 0, not inf.

    The guard is an exact comparison, not a tolerance: PETSc reciprocates
    1e-20 to 1e20, so a `PETSC_MACHINE_EPSILON` test here would diverge.
    """
    simdvec.reciprocal_inplace(x, n)


def VecExp_Seq(x: Ptr, n: Int):
    simdvec.exp_inplace(x, n)


def VecLog_Seq(x: Ptr, n: Int):
    """Upstream applies no domain guard: log(0) = -inf, log(neg) = nan."""
    simdvec.log_inplace(x, n)


def VecSqrtAbs_Seq(x: Ptr, n: Int):
    simdvec.sqrtabs_inplace(x, n)


def VecChop_Seq(x: Ptr, n: Int, tol: Float64):
    """x[i] = 0 where |x[i]| < tol, otherwise x[i] is left alone.

    rvector.c VecChop.  This zeroes the small entries; it does *not* replace the
    survivors with +/-1.  The comparison is a strict `<` against `tol`, so an
    entry whose magnitude is exactly `tol` survives -- which is why
    `chop(0.0)` is a no-op.
    """
    simdvec.chop_inplace(x, n, tol)


def VecAbs_Seq(x: Ptr, n: Int):
    simdvec.abs_inplace(x, n)


def VecConjugate_Seq(x: Ptr, n: Int):
    """PetscConj is the identity for a real build; the loop stays because
    upstream keeps it and it is where a complex build would differ."""
    simdvec.conjugate_inplace(x, n)


def VecCopy_Seq(x: Ptr, y: Ptr, n: Int):
    simdvec.copy(y, x, n)


def VecSwap_Seq(x: Ptr, y: Ptr, n: Int):
    simdvec.swap_inplace(x, y, n)


def VecSet_Seq(x: Ptr, alpha: Float64, n: Int):
    if alpha == 0.0:
        VecZeroEntries_Seq(x, n)
    else:
        simdvec.set_const(x, n, alpha)


def VecZeroEntries_Seq(x: Ptr, n: Int):
    simdvec.zero(x, n)


def VecScale_Seq(x: Ptr, alpha: Float64, n: Int):
    simdvec.map1(x, x, n, alpha)


def VecShift_Seq(x: Ptr, alpha: Float64, n: Int):
    simdvec.shift(x, x, n, alpha)


def VecAXPY_Seq(y: Ptr, alpha: Float64, x: Ptr, n: Int):
    """y = alpha*x + y.  Handles the x aliases y case, as upstream does."""
    # the two branches are the same expression once x is y, so there is one
    # sweep; the alias is still what upstream means and it reads correctly
    simdvec.axpy(y, x, n, alpha)


def VecAYPX_Seq(y: Ptr, alpha: Float64, x: Ptr, n: Int):
    """y = x + alpha*y, with upstream's three special-cased alpha values."""
    if alpha == 0.0:
        VecCopy_Seq(x, y, n)
    elif alpha == 1.0:
        VecAXPY_Seq(y, alpha, x, n)
    else:
        # covers alpha == -1.0 as alpha*y with alpha == -1.0, exactly as the
        # subtraction branch upstream writes it
        simdvec.aypx(y, x, n, alpha)


def VecAXPBY_Seq(y: Ptr, alpha: Float64, beta: Float64, x: Ptr, n: Int):
    """y = alpha*x + beta*y.

    `beta == 0` replaces rather than accumulates, so it is a scale-and-set,
    not a `VecAXPY` onto the old `y`; `alpha == 0` reduces to `VecScale`.
    """
    if beta == 0.0:
        simdvec.map1(y, x, n, alpha)
    elif alpha == 0.0:
        VecScale_Seq(y, beta, n)
    else:
        simdvec.axpby(y, x, n, alpha, beta)


def VecAXPBYPCZ_Seq(
    z: Ptr, alpha: Float64, beta: Float64, gamma: Float64, x: Ptr, y: Ptr, n: Int
):
    """z = alpha*x + beta*y + gamma*z.  dvecimpl.h VecAXPBYPCZ_Seq."""
    simdvec.axpbypcz(z, x, y, n, alpha, beta, gamma)


def VecWAXPY_Seq(w: Ptr, alpha: Float64, x: Ptr, y: Ptr, n: Int):
    """w = alpha*x + y.  Upstream fuses the trailing add, so w must not alias."""
    simdvec.waxpy(w, x, y, n, alpha)


def VecTDot_Seq(x: Ptr, y: Ptr, val: Ptr, n: Int):
    """val = y^T x.  rvector.c VecTDot: the indefinite dot that skips the
    conjugate, so for a real build it is BLASdot with the arguments swapped."""
    val.unsafe_store(0, simdvec.dot(y, x, n))


def VecMAXPY_Seq(y: Ptr, nv: Int, alpha: Ptr, x: PI, n: Int):
    """y = y + sum_k alpha[k]*x[k].  `x` is an array of vector addresses."""
    var k = 0
    while k < nv:
        simdvec.axpy(y, Ptr(unsafe_from_address=x.unsafe_load(k)), n,
                     alpha.unsafe_load(k))
        k += 1


def VecMaxPointwiseDivide_Seq(x: Ptr, y: Ptr, rval: Ptr, n: Int):
    """max_i |x_i / y_i|.  Where y_i is exactly zero upstream substitutes x_i
    rather than skipping the entry; reproduced here."""
    rval.unsafe_store(0, simdvec.max_pointwise_divide(x, y, n))


# ====================================================================
#  petsc4py/PETSc/Mat.pyx
# ====================================================================
#
# A sequential AIJ matrix is compressed sparse row: `irow[nrow + 1]` row starts,
# `icol[nnz]` column indices ascending within a row, `data[nnz]` values in the
# same order.  This is the layout `MATSEQAIJ` uses, so a `setValues` on one side
# and a `multVec` on the other line up entry for entry.


def MatGetDiagonalMarkers_SeqAIJ(
    irow: PI, icol: PI, diag: PI, n: Int
) -> Int:
    """aij.c MatGetDiagonalMarkers_SeqAIJ.

    `diag[i]` is the offset of a_ii inside row i, or the row end when the
    diagonal is absent.  Returns the upstream `diagDense` flag.
    """
    var diag_dense = 1
    var i = 0
    while i < n:
        var bl = irow.unsafe_load(i)
        var be = irow.unsafe_load(i + 1)
        diag.unsafe_store(i, be)
        var j = bl
        while j < be:
            if icol.unsafe_load(j) == i:
                diag.unsafe_store(i, j)
                break
            j += 1
        if diag.unsafe_load(i) == be:
            diag_dense = 0
        i += 1
    return diag_dense


def MatMult_SeqAIJ(x: Ptr, y: Ptr, irow: PI, icol: PI, data: Ptr, nrow: Int):
    """y = A x.  aij.c MatMult_SeqAIJ_A_Private."""
    var i = 0
    while i < nrow:
        var sum = 0.0
        var j = irow.unsafe_load(i)
        var je = irow.unsafe_load(i + 1)
        while j < je:
            sum += data.unsafe_load(j) * x.unsafe_load(icol.unsafe_load(j))
            j += 1
        y.unsafe_store(i, sum)
        i += 1


def MatMultAdd_SeqAIJ(x: Ptr, y: Ptr, irow: PI, icol: PI, data: Ptr, nrow: Int):
    """y = A x + y.  aij.c MatMultAdd_SeqAIJ_A_Private with beta = 1."""
    var i = 0
    while i < nrow:
        var sum = y.unsafe_load(i)
        var j = irow.unsafe_load(i)
        var je = irow.unsafe_load(i + 1)
        while j < je:
            sum += data.unsafe_load(j) * x.unsafe_load(icol.unsafe_load(j))
            j += 1
        y.unsafe_store(i, sum)
        i += 1


def MatMultTranspose_SeqAIJ(
    x: Ptr, y: Ptr, irow: PI, icol: PI, data: Ptr, nrow: Int, ncols: Int
):
    """y = Aᵀ x.  aij.c MatMultTranspose_SeqAIJ_A_Private.

    The output is indexed by column, so it is `ncols` long and only that many
    entries are cleared; `nrow` is how many rows of `icol` are visited.  The
    sequential kernel is only ever reached with beta = 1 from the Krylov
    drivers, so the existing y is accumulated into rather than replaced.
    """
    var j = 0
    while j < ncols:
        y.unsafe_store(j, 0.0)
        j += 1
    var i = 0
    while i < nrow:
        var jb = irow.unsafe_load(i)
        var je = irow.unsafe_load(i + 1)
        var xi = x.unsafe_load(i)
        while jb < je:
            var c = icol.unsafe_load(jb)
            y.unsafe_store(c, y.unsafe_load(c) + data.unsafe_load(jb) * xi)
            jb += 1
        i += 1


def MatGetDiagonal_SeqAIJ(diag: Ptr, irow: PI, icol: PI, data: Ptr, nrow: Int):
    """d = diag(A).  A missing diagonal entry stays at zero, as upstream."""
    var i = 0
    while i < nrow:
        var v = 0.0
        var j = irow.unsafe_load(i)
        var je = irow.unsafe_load(i + 1)
        while j < je:
            if icol.unsafe_load(j) == i:
                v = data.unsafe_load(j)
                break
            j += 1
        diag.unsafe_store(i, v)
        i += 1


def MatScale_SeqAIJ(data: Ptr, nnz: Int, alpha: Float64):
    var j = 0
    while j < nnz:
        data.unsafe_store(j, data.unsafe_load(j) * alpha)
        j += 1


def MatZeroEntries_SeqAIJ(data: Ptr, nnz: Int):
    var j = 0
    while j < nnz:
        data.unsafe_store(j, 0.0)
        j += 1


# ====================================================================
#  petsc4py/PETSc/PC.pyx
# ====================================================================

comptime PC_NONE: Int = 0
comptime PC_JACOBI: Int = 2
comptime PC_SOR: Int = 4
comptime PC_ILU: Int = 5
comptime PC_ICC: Int = 6

# petscpc.h PCJACOBIType
comptime PC_JACOBI_DIAGONAL: Int = 0
comptime PC_JACOBI_ROWMAX: Int = 1
comptime PC_JACOBI_ROWL1: Int = 2
comptime PC_JACOBI_ROWSUM: Int = 3

# petscmat.h MatSORType
comptime SOR_ZERO_INITIAL_GUESS: Int = 16
comptime SOR_FORWARD_SWEEP: Int = 1
comptime SOR_BACKWARD_SWEEP: Int = 2
comptime SOR_LOCAL_FORWARD_SWEEP: Int = 4
comptime SOR_LOCAL_BACKWARD_SWEEP: Int = 8
comptime SOR_LOCAL_SYMMETRIC_SWEEP: Int = 12


def _JacobiFill(
    dst: Ptr, jtype: Int, arow: PI, acol: PI, adata: Ptr, nrow: Int
):
    """jacobi.c PCSetUp_Jacobi, the `switch (jac->type)` that fills either the
    diagonal vector or the diagonal-square-root vector."""
    var i = 0
    if jtype == PC_JACOBI_DIAGONAL:
        MatGetDiagonal_SeqAIJ(dst, arow, acol, adata, nrow)
    elif jtype == PC_JACOBI_ROWMAX:
        MatGetRowMaxAbs_SeqAIJ(dst, arow, acol, adata, nrow)
    elif jtype == PC_JACOBI_ROWL1:
        MatGetRowSumAbs_SeqAIJ(dst, arow, acol, adata, nrow)
    else:
        MatGetRowSum_SeqAIJ(dst, arow, acol, adata, nrow)


def PCSetUp_Jacobi(
    diag: Ptr, jtype: Int, useabs: Int, fixdiag: Int, scale: Float64,
    arow: PI, acol: PI, adata: Ptr, nrow: Int, didx: Ptr
):
    """jacobi.c PCSetUp_Jacobi, the `jac->diag` branch, reached from
    PCSetUp_Jacobi_NonSymmetric and so from PCApply_Jacobi."""
    var i = 0
    _JacobiFill(diag, jtype, arow, acol, adata, nrow)
    if jtype == PC_JACOBI_ROWL1:
        # upstream only sign-fixes when the matrix is not known SPD, which the
        # sequential PC has no way to know here, so the flip always applies
        MatGetDiagonal_SeqAIJ(didx, arow, acol, adata, nrow)
        i = 0
        while i < nrow:
            if didx.unsafe_load(i) < 0.0:
                diag.unsafe_store(i, -diag.unsafe_load(i))
            i += 1
        if scale != 1.0:
            # diag <- scale*(rowl1 - true_diag) + true_diag
            i = 0
            while i < nrow:
                diag.unsafe_store(
                    i,
                    scale * (diag.unsafe_load(i) - didx.unsafe_load(i))
                    + didx.unsafe_load(i),
                )
                i += 1
    VecReciprocal_Seq(diag, nrow)
    if useabs:
        VecAbs_Seq(diag, nrow)
    if fixdiag:
        # the zero test is on the inverse, i.e. after VecReciprocal
        i = 0
        while i < nrow:
            if diag.unsafe_load(i) == 0.0:
                diag.unsafe_store(i, 1.0)
            i += 1


def PCSetUp_Jacobi_Symmetric(
    diagsqrt: Ptr, jtype: Int, arow: PI, acol: PI, adata: Ptr, nrow: Int
):
    """jacobi.c PCSetUp_Jacobi, the `jac->diagsqrt` branch, reached from
    PCSetUp_Jacobi_Symmetric and so from applySymmetricLeft/Right_Jacobi.

    None of useabs, fixdiag, scale or the rowl1 sign fix apply on this path.
    """
    _JacobiFill(diagsqrt, jtype, arow, acol, adata, nrow)
    var i = 0
    while i < nrow:
        var d = diagsqrt.unsafe_load(i)
        if d < 0.0:
            diagsqrt.unsafe_store(i, 1.0 / sqrt(abs(-d)))
        elif d > 0.0:
            diagsqrt.unsafe_store(i, 1.0 / sqrt(abs(d)))
        else:
            diagsqrt.unsafe_store(i, 1.0)
        i += 1


def PCApply_Jacobi(y: Ptr, x: Ptr, diag: Ptr, n: Int):
    """jacobi.c PCApply_Jacobi: a single VecPointwiseMult(y, x, diag)."""
    simdvec.pointwise_mult(y, x, diag, n)


def MatGetRowSum_SeqAIJ(diag: Ptr, arow: PI, acol: PI, adata: Ptr, nrow: Int):
    var i = 0
    while i < nrow:
        var s = 0.0
        var j = arow.unsafe_load(i)
        var je = arow.unsafe_load(i + 1)
        while j < je:
            s += adata.unsafe_load(j)
            j += 1
        diag.unsafe_store(i, s)
        i += 1


def MatGetRowSumAbs_SeqAIJ(diag: Ptr, arow: PI, acol: PI, adata: Ptr, nrow: Int):
    var i = 0
    while i < nrow:
        var s = 0.0
        var j = arow.unsafe_load(i)
        var je = arow.unsafe_load(i + 1)
        while j < je:
            s += abs(adata.unsafe_load(j))
            j += 1
        diag.unsafe_store(i, s)
        i += 1


def MatGetRowMaxAbs_SeqAIJ(diag: Ptr, arow: PI, acol: PI, adata: Ptr, nrow: Int):
    var i = 0
    while i < nrow:
        var s = 0.0
        var j = arow.unsafe_load(i)
        var je = arow.unsafe_load(i + 1)
        while j < je:
            var v = abs(adata.unsafe_load(j))
            s = v if v > s else s
            j += 1
        diag.unsafe_store(i, s)
        i += 1


def MatInvertDiagonalForSOR_SeqAIJ(
    idiag: Ptr, mdiag: Ptr, dmark: PI, arow: PI, acol: PI, adata: Ptr,
    omega: Float64, fshift: Float64, nrow: Int
):
    """aij.c MatInvertDiagonalForSOR_SeqAIJ.

    omega == 1 and fshift <= 0 takes upstream's plain `1/a_ii` branch; every
    other combination uses `omega/(fshift + a_ii)`.
    """
    if omega == 1.0 and fshift <= 0.0:
        var i = 0
        while i < nrow:
            mdiag.unsafe_store(i, adata.unsafe_load(dmark.unsafe_load(i)))
            idiag.unsafe_store(i, 1.0 / adata.unsafe_load(dmark.unsafe_load(i)))
            i += 1
    else:
        var i = 0
        while i < nrow:
            mdiag.unsafe_store(i, adata.unsafe_load(dmark.unsafe_load(i)))
            idiag.unsafe_store(i, omega / (fshift + adata.unsafe_load(dmark.unsafe_load(i))))
            i += 1


def PCApply_SOR(
    x: Ptr, y: Ptr, arow: PI, acol: PI, adata: Ptr, dmark: PI,
    idiag: Ptr, mdiag: Ptr, t: Ptr, omega: Float64, fshift: Float64, its: Int,
    lits: Int, flag: Int, nrow: Int
):
    """sorc.c PCApply_SOR -> aij.c MatSOR_SeqAIJ, one `its * lits` sweep block.

    The default PCSOR configuration is a symmetric local sweep with
    SOR_ZERO_INITIAL_GUESS, which performs exactly one symmetric sweep in the
    zero-guess prologue and then leaves `its` at zero, so the main loop does
    not run.  That structure is kept verbatim.
    """
    var i = its * lits
    MatInvertDiagonalForSOR_SeqAIJ(
        idiag, mdiag, dmark, arow, acol, adata, omega, fshift, nrow
    )

    # zero-initial-guess prologue
    var xb = t
    if flag & (SOR_FORWARD_SWEEP | SOR_LOCAL_FORWARD_SWEEP):
        i = 0
        while i < nrow:
            var sum = x.unsafe_load(i)
            var j = arow.unsafe_load(i)
            var jend = dmark.unsafe_load(i)
            while j < jend:
                sum -= adata.unsafe_load(j) * y.unsafe_load(acol.unsafe_load(j))
                j += 1
            t.unsafe_store(i, sum)
            y.unsafe_store(i, sum * idiag.unsafe_load(i))
            i += 1
    if flag & (SOR_BACKWARD_SWEEP | SOR_LOCAL_BACKWARD_SWEEP):
        i = nrow - 1
        while i >= 0:
            var sum = xb.unsafe_load(i)
            var j = dmark.unsafe_load(i) + 1
            var jend = arow.unsafe_load(i + 1)
            while j < jend:
                sum -= adata.unsafe_load(j) * y.unsafe_load(acol.unsafe_load(j))
                j += 1
            y.unsafe_store(
                i, (1.0 - omega) * y.unsafe_load(i) + sum * idiag.unsafe_load(i)
            )
            i -= 1
    i -= 1

    # main loop
    while i > 0:
        i -= 1
        if flag & (SOR_FORWARD_SWEEP | SOR_LOCAL_FORWARD_SWEEP):
            var k = 0
            while k < nrow:
                # lower
                var sum = x.unsafe_load(k)
                var j = arow.unsafe_load(k)
                var jend = dmark.unsafe_load(k)
                while j < jend:
                    sum -= adata.unsafe_load(j) * y.unsafe_load(acol.unsafe_load(j))
                    j += 1
                t.unsafe_store(k, sum)
                # upper
                j = dmark.unsafe_load(k) + 1
                jend = arow.unsafe_load(k + 1)
                while j < jend:
                    sum -= adata.unsafe_load(j) * y.unsafe_load(acol.unsafe_load(j))
                    j += 1
                y.unsafe_store(
                    k, (1.0 - omega) * y.unsafe_load(k) + sum * idiag.unsafe_load(k)
                )
                k += 1
        if flag & (SOR_BACKWARD_SWEEP | SOR_LOCAL_BACKWARD_SWEEP):
            var k = nrow - 1
            while k >= 0:
                var sum = t.unsafe_load(k)
                var j = dmark.unsafe_load(k) + 1
                var jend = arow.unsafe_load(k + 1)
                while j < jend:
                    sum -= adata.unsafe_load(j) * y.unsafe_load(acol.unsafe_load(j))
                    j += 1
                y.unsafe_store(
                    k, (1.0 - omega) * y.unsafe_load(k) + sum * idiag.unsafe_load(k)
                )
                k -= 1



# --- incomplete factorisation ---------------------------------------------
# matimpl.h: the sorted linked list with per-entry fill levels that both the
# ILU and the IC symbolic factorizations build their sparsity pattern with.


def LLInsertLocation(
    assume_sorted: Int, k: Int, idx_start: Int, entry: Int, nlnk: Int,
    lnkdata: Int, lnk: PI
) -> Tuple[Int, Int]:
    """matimpl.h PetscLLInsertLocation_Private.  Returns (nlnk, lnkdata)."""
    var ld = lnkdata
    if not assume_sorted and k != 0 and entry < ld:
        ld = idx_start
    var location = ld
    var walking = True
    while walking:
        # the list is kept sorted, so walk while the successor is smaller
        ld = lnk.unsafe_load(location)
        if entry > ld:
            location = ld
        else:
            walking = False
    lnk.unsafe_store(location, entry)
    lnk.unsafe_store(entry, ld)
    return (nlnk + 1, entry)


def _BTLookupSet(bt: PI, entry: Int) -> Int:
    """PetscBTLookupSet: report whether `entry` is already marked, then mark it.

    Marking is what keeps the list duplicate-free; testing alone lets the same
    column be spliced in twice, which corrupts the chain.
    """
    var old = bt.unsafe_load(entry)
    bt.unsafe_store(entry, 1)
    return old


def IncompleteLLInit(
    nidx: Int, idx: PI, idx_start: Int, perm: PI, lnk: PI, lnklvl: PI,
    bt: PI
) -> Int:
    """matimpl.h PetscIncompleteLLInit.  New entries enter at level 0."""
    var nlnk = 0
    var lnkdata = idx_start
    var k = 0
    while k < nidx:
        var entry = perm.unsafe_load(idx.unsafe_load(k))
        if _BTLookupSet(bt, entry) == 0:
            var r = LLInsertLocation(0, k, idx_start, entry, nlnk, lnkdata, lnk)
            nlnk = r[0]
            lnkdata = r[1]
            lnklvl.unsafe_store(entry, 0)
        k += 1
    return nlnk


def IncompleteLLAdd(
    nidx: Int, idx: PI, level: Float64, idxlvl: PI, idx_start: Int, lnk: PI,
    lnklvl: PI, bt: PI, prow_offset: Int, assume_sorted: Int
) -> Int:
    """matimpl.h PetscIncompleteLLAdd_Private.

    The fill level of an entry is capped at `level`; an entry already on the
    list keeps whichever of the two candidate levels is smaller.
    """
    var nlnk = 0
    var lnkdata = idx_start
    var k = 0
    while k < nidx:
        var incrlev = idxlvl.unsafe_load(k) + prow_offset + 1
        if Float64(incrlev) <= level:
            var entry = idx.unsafe_load(k)
            if _BTLookupSet(bt, entry) == 0:
                var r = LLInsertLocation(
                    assume_sorted, k, idx_start, entry, nlnk, lnkdata, lnk
                )
                nlnk = r[0]
                lnkdata = r[1]
                lnklvl.unsafe_store(entry, incrlev)
            elif lnklvl.unsafe_load(entry) > incrlev:
                lnklvl.unsafe_store(entry, incrlev)
        k += 1
    return nlnk


def IncompleteLLClean(
    idx_start: Int, lnk_max: Int, nlnk: Int, lnk: PI, lnklvl: PI, indices: PI,
    indiceslvl: PI, bt: PI
):
    """matimpl.h PetscIncompleteLLClean: drain the list, then reset it."""
    var idx = idx_start
    var j = 0
    while j < nlnk:
        idx = lnk.unsafe_load(idx)
        indices.unsafe_store(j, idx)
        indiceslvl.unsafe_store(j, lnklvl.unsafe_load(idx))
        lnklvl.unsafe_store(idx, -1)
        bt.unsafe_store(idx, 0)
        j += 1
    lnk.unsafe_store(idx_start, lnk_max)


def ILUApplyFactor_SeqAIJ(
    b: Ptr, x: Ptr, tmp: Ptr, bi: PI, bj: PI, ba: Ptr, bdiag: PI, r: PI,
    c: PI, nrow: Int
):
    """aijfact.c MatSolve_SeqAIJ, the triangular solve against an ILU factor.

    The factor stores the L rows first and the U rows in reverse, with each
    diagonal of U already inverted; `r` and `c` are the row/column orderings.
    """
    if not nrow:
        return
    tmp.unsafe_store(0, b.unsafe_load(r.unsafe_load(0)))
    var i = 1
    while i < nrow:
        var sum = b.unsafe_load(r.unsafe_load(i))
        var j = bi.unsafe_load(i)
        var jend = bi.unsafe_load(i + 1)
        while j < jend:
            sum -= ba.unsafe_load(j) * tmp.unsafe_load(bj.unsafe_load(j))
            j += 1
        tmp.unsafe_store(i, sum)
        i += 1

    i = nrow - 1
    while i >= 0:
        var v = bdiag.unsafe_load(i + 1) + 1
        var nz = bdiag.unsafe_load(i) - bdiag.unsafe_load(i + 1) - 1
        var sum = tmp.unsafe_load(i)
        var j = 0
        while j < nz:
            sum -= ba.unsafe_load(v + j) * tmp.unsafe_load(bj.unsafe_load(v + j))
            j += 1
        sum = sum * ba.unsafe_load(v + nz)
        tmp.unsafe_store(i, sum)
        x.unsafe_store(c.unsafe_load(i), sum)
        i -= 1


def ICCApplyFactor_SeqAIJ(
    b: Ptr, x: Ptr, y: Ptr, ui: PI, uj: PI, ua: Ptr, udiag: PI, r: PI,
    tirow: PI, ticol: PI, tdata: Ptr, nrow: Int
) -> Int:
    """Solve with the IC(levels) factor, which PETSc stores as L*D*Lᵀ with only
    the upper triangle present: the strict upper slot (i, k) holds -L[k, i] and
    the diagonal slot holds 1/d_i, the reciprocal of the pivot.

    The two sweeps are the forward solve for the unit-lower L and the backward
    solve for Lᵀ with the diagonal divided in between, as
    `baij/seq/baijsolvnat1.c` does.  L's strictly-lower part is the transpose of
    the stored upper triangle, so the forward sweep walks the transpose
    pattern `tirow`/`ticol`/`tdata` that the setup builds alongside the factor.
    With a permutation the solve happens in the permuted index space, and `r`
    maps back.
    """
    if not nrow:
        return 0
    # forward solve the unit-lower factor: y[i] = b[i] - sum_k L[i,k] y[k]
    var i = 0
    while i < nrow:
        var sum = b.unsafe_load(r.unsafe_load(i))
        var j = tirow.unsafe_load(i)
        var jend = tirow.unsafe_load(i + 1)
        while j < jend:
            var k = ticol.unsafe_load(j)
            if k < i:
                # the stored value is -L[i, k]
                sum += tdata.unsafe_load(j) * y.unsafe_load(k)
            j += 1
        y.unsafe_store(i, sum)
        i += 1
    # backward solve Lᵀ: the diagonal enters the pivot only, so it scales y[i]
    # alone and not the accumulated strict-upper term
    i = nrow - 1
    while i >= 0:
        var sum = y.unsafe_load(i) * ua.unsafe_load(udiag.unsafe_load(i))
        var j = ui.unsafe_load(i)
        var jend = udiag.unsafe_load(i)
        while j < jend:
            sum += ua.unsafe_load(j) * x.unsafe_load(uj.unsafe_load(j))
            j += 1
        x.unsafe_store(r.unsafe_load(i), sum)
        i -= 1
    return 0


def MatILUFactorNumeric_SeqAIJ(
    ba: Ptr, rtmp: Ptr, arow: PI, acol: PI, adata: Ptr, bi: PI, bj: PI,
    bdiag: PI, r: PI, ic: PI, nrow: Int
) -> Int:
    """aijfact.c MatLUFactorNumeric_SeqAIJ, shift type `none`.

    The shift-and-retry machinery (`MatPivotCheck`, `MAT_SHIFT_NONZERO`,
    `MAT_SHIFT_POSITIVE_DEFINITE`) is not ported; a zero pivot propagates as
    upstream's would without a shift, which the KSP then reports as a NaN/inf
    residual.  Returns the number of factor entries written.
    """
    var i = 0
    while i < nrow:
        # zero rtmp
        var j = 0
        var jend = bi.unsafe_load(i + 1) - bi.unsafe_load(i)
        var jj = bi.unsafe_load(i)
        while j < jend:
            rtmp.unsafe_store(bj.unsafe_load(jj), 0.0)
            jj += 1
            j += 1
        # U part
        jend = bdiag.unsafe_load(i) - bdiag.unsafe_load(i + 1)
        jj = bdiag.unsafe_load(i + 1) + 1
        j = 0
        while j < jend:
            rtmp.unsafe_store(bj.unsafe_load(jj), 0.0)
            jj += 1
            j += 1
        # load in the initial (unfactored) row
        jend = arow.unsafe_load(r.unsafe_load(i) + 1) - arow.unsafe_load(r.unsafe_load(i))
        jj = arow.unsafe_load(r.unsafe_load(i))
        j = 0
        while j < jend:
            rtmp.unsafe_store(ic.unsafe_load(acol.unsafe_load(jj)), adata.unsafe_load(jj))
            jj += 1
            j += 1

        # elimination
        jj = bi.unsafe_load(i)
        var row = bj.unsafe_load(jj)
        jj += 1
        var nzL = bi.unsafe_load(i + 1) - bi.unsafe_load(i)
        var k = 0
        while k < nzL:
            var pc = rtmp.unsafe_load(row)
            if pc != 0.0:
                var pv = ba.unsafe_load(bdiag.unsafe_load(row))
                var multiplier = pc * pv
                rtmp.unsafe_store(row, multiplier)
                var pj = bdiag.unsafe_load(row + 1) + 1
                var nz = bdiag.unsafe_load(row) - bdiag.unsafe_load(row + 1) - 1
                j = 0
                while j < nz:
                    var c = bj.unsafe_load(pj + j)
                    rtmp.unsafe_store(c, rtmp.unsafe_load(c) - multiplier * ba.unsafe_load(pj + j))
                    j += 1
            row = bj.unsafe_load(jj)
            jj += 1
            k += 1

        # finished row, store into the factor
        jend = bi.unsafe_load(i + 1) - bi.unsafe_load(i)
        jj = bi.unsafe_load(i)
        j = 0
        while j < jend:
            ba.unsafe_store(jj, rtmp.unsafe_load(bj.unsafe_load(jj)))
            jj += 1
            j += 1
        jend = bdiag.unsafe_load(i) - bdiag.unsafe_load(i + 1) - 1
        jj = bdiag.unsafe_load(i + 1) + 1
        j = 0
        while j < jend:
            ba.unsafe_store(jj, rtmp.unsafe_load(bj.unsafe_load(jj)))
            jj += 1
            j += 1

        # mark the diagonal and invert it, as upstream does for simpler solves
        ba.unsafe_store(bdiag.unsafe_load(i), 1.0 / rtmp.unsafe_load(i))
        i += 1
    return bdiag.unsafe_load(0) + 1


def MatICCFactorNumeric_SeqAIJ(
    ba: Ptr, rtmp: Ptr, il: PI, c2r: PI, arow: PI, acol: PI, adata: Ptr,
    bi: PI, bj: PI, bdiag: PI, r: PI, riip: PI, nrow: Int
):
    """aijfact.c MatCholeskyFactorNumeric_SeqAIJ with shift type `none`.

    `il` and `c2r` carry the row-update bookkeeping: `il[i]` is the first
    nonzero of U(i,k:) and `c2r` chains the rows that touch a given column.
    """
    var i = 0
    while i < nrow:
        c2r.unsafe_store(i, nrow)
        i += 1
    if nrow:
        il.unsafe_store(0, 0)

    var k = 0
    while k < nrow:
        # zero rtmp
        var j = 0
        var jend = bi.unsafe_load(k + 1) - bi.unsafe_load(k)
        var jj = bi.unsafe_load(k)
        while j < jend:
            rtmp.unsafe_store(bj.unsafe_load(jj), 0.0)
            jj += 1
            j += 1
        # load the upper triangular part of the unfactored row
        var bval = bi.unsafe_load(k)
        j = arow.unsafe_load(r.unsafe_load(k))
        jend = arow.unsafe_load(r.unsafe_load(k) + 1)
        while j < jend:
            var col = riip.unsafe_load(acol.unsafe_load(j))
            if col >= k:
                rtmp.unsafe_store(col, adata.unsafe_load(j))
                ba.unsafe_store(bval, 0.0)
                bval += 1
            j += 1

        # add in the rows i with U(i,k) != 0
        var dk = rtmp.unsafe_load(k)
        i = c2r.unsafe_load(k)
        while i < k:
            var nexti = c2r.unsafe_load(i)
            var ili = il.unsafe_load(i)
            var uikdi = -ba.unsafe_load(ili) * ba.unsafe_load(bdiag.unsafe_load(i))
            dk += uikdi * ba.unsafe_load(ili)
            ba.unsafe_store(ili, uikdi)
            var jmin = ili + 1
            var jmax = bi.unsafe_load(i + 1)
            if jmin < jmax:
                j = jmin
                while j < jmax:
                    var c = bj.unsafe_load(j)
                    rtmp.unsafe_store(c, rtmp.unsafe_load(c) + uikdi * ba.unsafe_load(j))
                    j += 1
                il.unsafe_store(i, jmin)
                var cj = bj.unsafe_load(jmin)
                c2r.unsafe_store(i, c2r.unsafe_load(cj))
                c2r.unsafe_store(cj, i)
            i = nexti

        # copy the row into U(k,:)
        var jmin = bi.unsafe_load(k)
        var jmax = bi.unsafe_load(k + 1) - 1
        if jmin < jmax:
            j = jmin
            while j < jmax:
                var col = bj.unsafe_load(j)
                ba.unsafe_store(j, rtmp.unsafe_load(col))
                j += 1
            il.unsafe_store(k, jmin)
            var cj = bj.unsafe_load(jmin)
            c2r.unsafe_store(k, c2r.unsafe_load(cj))
            c2r.unsafe_store(cj, k)
        ba.unsafe_store(bdiag.unsafe_load(k), 1.0 / dk)
        k += 1

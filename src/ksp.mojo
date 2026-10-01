"""Krylov drivers, in the order they appear in `petsc4py/PETSc/KSP.pyx`.

Each routine is a transcription of the PETSc C solver the corresponding
`KSP.Type` selects: `KSP.Type.CG` -> `KSPSolve_CG` (src/ksp/ksp/impls/cg/cg.c),
`Type.GMRES` -> `KSPSolve_GMRES` + `KSPGMRESCycle` + `KSPGMRESUpdateHessenberg`
+ `KSPGMRESBuildSoln` (src/ksp/ksp/impls/gmres/gmres.c, borthog.c, borthog2.c),
`Type.FGMRES` -> the fgmres.c equivalents, and so on.

Mojo cannot hold PETSc's `KSP` object, so its scalar fields travel in
caller-owned blocks, laid out here and mirrored exactly by
`python/mojopetsc4py/PETSc/KSP.py`:

    cfg  (Float64)   0 rtol           4 breakdowntol   8  gmres rnorm0
                    1 atol           5 rnorm0         9+ residual history
                    2 divtol         6 ttol           12 RHS norm
                    3 haptol         7 rnorm (out)    13 unused
                                                         14 seeded flag

    cfgi (Int)       0 normtype       5 convmaxits     10 its (output)
                    1 pc_side        6 max_it         11 reason (output)
                    2 guess_zero     7 max_k          12 seed from RHS
                    3 min_it         8 orthog
                    4 chknorm        9 cgstype

    pci  (Int)       0 PC type        4 aux3           8 aux7
                    1 aux0           5 aux4           9 aux8
                    2 aux1           6 aux5           10 PCSOR sweep flag
                    3 aux2           7 aux6

    pcf  (Float64)   0 omega, 1 fshift   (PCSOR only)

The GMRES-family scalar block `hs` is one allocation with fixed offsets:

    HH        (max_k + 2) * (max_k + 1), column major
    HES       (max_k + 1) * (max_k + 1), column major
    CC        max_k + 1
    SS        max_k + 1
    RS        max_k + 2
    ORTHOG    max_k + 2

`vv` is the Arnoldi basis, (max_k + 1) vectors of length n laid out contiguously;
`prevec` is FGMRES's copy of the preconditioned directions, same shape.
"""

from ported import *

comptime KSP_CG: Int = 2
comptime KSP_BCGS: Int = 6
comptime KSP_GMRES: Int = 12
comptime KSP_FGMRES: Int = 40

comptime KSP_NORM_DEFAULT: Int = 0
comptime KSP_NORM_NONE: Int = 1
comptime KSP_NORM_PRECONDITIONED: Int = 2
comptime KSP_NORM_UNPRECONDITIONED: Int = 3
comptime KSP_NORM_NATURAL: Int = 4

comptime CGS_REFINE_NEVER: Int = 0
comptime CGS_REFINE_IFNEEDED: Int = 1
comptime CGS_REFINE_ALWAYS: Int = 2
comptime ORTHOG_CGS: Int = 0
comptime ORTHOG_MGS: Int = 1

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

comptime PC_LEFT: Int = 0
comptime PC_RIGHT: Int = 1
comptime PC_SYMMETRIC: Int = 2

comptime INF: Float64 = 1.7976931348623157e308


def KSPConvergedDefault(n: Int, rnorm: Float64, cfg: Ptr, cfgi: PI) -> Int:
    """iterativ.c KSPConvergedDefault.

    The one-time seeding of `rnorm0` and `ttol` stays here, in upstream's
    place.  Its `!guess_zero` branch needs a preconditioner application on the
    right-hand side, which the kernel does not own, so the wrapper computes
    that norm into `cfg[12]` and flags `cfgi[12]`; the `guess_zero` branch
    takes `rnorm` as upstream does.  Everything else is verbatim, including the
    `convmaxits` short circuit that reports KSP_CONVERGED_ITS rather than a
    divergence.
    """
    var normtype = cfgi.unsafe_load(0)
    var min_it = cfgi.unsafe_load(3)
    var chknorm = cfgi.unsafe_load(4)
    var convmaxits = cfgi.unsafe_load(5)
    var max_it = cfgi.unsafe_load(6)

    if convmaxits and n >= max_it:
        return KSP_CONVERGED_ITS
    if normtype == KSP_NORM_NONE:
        return KSP_CONVERGED_ITERATING

    var rtol = cfg.unsafe_load(0)
    var atol = cfg.unsafe_load(1)
    var divtol = cfg.unsafe_load(2)

    if not n and cfg.unsafe_load(14) == 0.0:
        var rnorm0 = rnorm
        if cfgi.unsafe_load(12) != 0:
            # a nonzero initial guess measures the right-hand side instead
            var snorm = cfg.unsafe_load(12)
            if not snorm:
                # special case: nonzero guess with a zero right-hand side
                snorm = rnorm
            rnorm0 = snorm
        cfg.unsafe_store(5, rnorm0)
        var ttol = rtol * rnorm0
        cfg.unsafe_store(6, ttol if ttol > atol else atol)
        cfg.unsafe_store(14, 1.0)

    if n <= chknorm:
        return KSP_CONVERGED_ITERATING
    if rnorm != rnorm or abs(rnorm) == INF:
        return KSP_DIVERGED_NANORINF
    if n < min_it:
        return KSP_CONVERGED_ITERATING

    var rnorm0 = cfg.unsafe_load(5)
    var ttol = cfg.unsafe_load(6)

    if rnorm <= ttol:
        if rnorm < atol:
            return KSP_CONVERGED_ATOL
        return KSP_CONVERGED_RTOL
    if rnorm >= divtol * rnorm0:
        return KSP_DIVERGED_DTOL
    return KSP_CONVERGED_ITERATING


def KSP_PCApply(pci: PI, pcf: Ptr, pstmp: Ptr, n: Int, x: Ptr, y: Ptr):
    """kspimpl.h KSP_PCApply -> precon.c PCApply, dispatched on the PC type."""
    var kind = pci.unsafe_load(0)
    if kind == PC_JACOBI:
        PCApply_Jacobi(y, x, Ptr(unsafe_from_address=pci.unsafe_load(1)), n)
    elif kind == PC_ILU:
        ILUApplyFactor_SeqAIJ(
            x, y, pstmp,
            PI(unsafe_from_address=pci.unsafe_load(1)),
            PI(unsafe_from_address=pci.unsafe_load(2)),
            Ptr(unsafe_from_address=pci.unsafe_load(3)),
            PI(unsafe_from_address=pci.unsafe_load(4)),
            PI(unsafe_from_address=pci.unsafe_load(5)),
            PI(unsafe_from_address=pci.unsafe_load(6)), n
        )
    elif kind == PC_ICC:
        _ = ICCApplyFactor_SeqAIJ(
            x, y, pstmp,
            PI(unsafe_from_address=pci.unsafe_load(1)),
            PI(unsafe_from_address=pci.unsafe_load(2)),
            Ptr(unsafe_from_address=pci.unsafe_load(3)),
            PI(unsafe_from_address=pci.unsafe_load(4)),
            PI(unsafe_from_address=pci.unsafe_load(5)),
            PI(unsafe_from_address=pci.unsafe_load(6)),
            PI(unsafe_from_address=pci.unsafe_load(7)),
            Ptr(unsafe_from_address=pci.unsafe_load(8)), n
        )
    elif kind == PC_SOR:
        # aux0..3 the matrix, aux4/aux5 the cached inverse and raw diagonal,
        # aux6 the length-n checkpoint upstream keeps in a->ssor_work
        PCApply_SOR(
            x, y,
            PI(unsafe_from_address=pci.unsafe_load(1)),
            PI(unsafe_from_address=pci.unsafe_load(2)),
            Ptr(unsafe_from_address=pci.unsafe_load(3)),
            PI(unsafe_from_address=pci.unsafe_load(4)),
            Ptr(unsafe_from_address=pci.unsafe_load(5)),
            Ptr(unsafe_from_address=pci.unsafe_load(6)),
            Ptr(unsafe_from_address=pci.unsafe_load(7)),
            pcf.unsafe_load(0), pcf.unsafe_load(1), 1, 1,
            pci.unsafe_load(10), n
        )
    else:
        VecCopy_Seq(x, y, n)


def PCApplySymmetric(pci: PI, pcf: Ptr, pstmp: Ptr, n: Int, x: Ptr, y: Ptr):
    """precon.c PCApplySymmetricLeft/Right.  Only the Jacobi diagonal square
    root is a symmetric preconditioner here; the others fall back to PCApply,
    which is what upstream does when `applysymmetricleft` is not set."""
    if pci.unsafe_load(0) == PC_JACOBI:
        PCApply_Jacobi(y, x, Ptr(unsafe_from_address=pci.unsafe_load(5)), n)
    else:
        KSP_PCApply(pci, pcf, pstmp, n, x, y)


def KSPUnwindPreconditioner(
    pci: PI, pcf: Ptr, pstmp: Ptr, pc_side: Int, n: Int, vsoln: Ptr, vt1: Ptr
):
    """itres.c KSPUnwindPreconditioner, without the diagonal-scaling branches."""
    VecCopy_Seq(vsoln, vt1, n)
    if pc_side == PC_RIGHT:
        KSP_PCApply(pci, pcf, pstmp, n, vsoln, vt1)
        VecCopy_Seq(vt1, vsoln, n)


def KSPInitialResidual(
    pci: PI, pcf: Ptr, pstmp: Ptr, pc_side: Int, guess_zero: Int,
    arow: PI, acol: PI, adata: Ptr, n: Int,
    vsoln: Ptr, vt1: Ptr, vt2: Ptr, vres: Ptr, vb: Ptr
):
    """itres.c KSPInitialResidual.

    The `!guess_zero` branch forms `r = b - A x` first and only then applies the
    preconditioner, so with `PC_RIGHT` the residual handed to the driver is
    unpreconditioned.
    """
    if not guess_zero:
        MatMult_SeqAIJ(vsoln, vt1, arow, acol, adata, n)
        VecCopy_Seq(vb, vt2, n)
        VecAXPY_Seq(vt2, -1.0, vt1, n)
    else:
        VecCopy_Seq(vb, vt2, n)
    if pc_side == PC_SYMMETRIC:
        PCApplySymmetric(pci, pcf, pstmp, n, vt2, vres)
    elif pc_side == PC_LEFT:
        KSP_PCApply(pci, pcf, pstmp, n, vt2, vres)
    else:
        VecCopy_Seq(vt2, vres, n)


def KSP_PCApplyBAorAB(
    pci: PI, pcf: Ptr, pstmp: Ptr, pc_side: Int,
    arow: PI, acol: PI, adata: Ptr, n: Int, x: Ptr, y: Ptr, w: Ptr
):
    """kspimpl.h KSP_PCApplyBAorAB -> precon.c PCApplyBAorAB, no diagonal scale.

    `PC_RIGHT` forms A·(M⁻¹x) with `w` as scratch, `PC_LEFT` forms (M⁻¹A)x, and
    `PC_SYMMETRIC` composes the two symmetric halves around the operator.
    """
    if pc_side == PC_RIGHT:
        KSP_PCApply(pci, pcf, pstmp, n, x, w)
        MatMult_SeqAIJ(w, y, arow, acol, adata, n)
    elif pc_side == PC_LEFT:
        MatMult_SeqAIJ(x, w, arow, acol, adata, n)
        KSP_PCApply(pci, pcf, pstmp, n, w, y)
    else:
        PCApplySymmetric(pci, pcf, pstmp, n, x, w)
        MatMult_SeqAIJ(w, y, arow, acol, adata, n)
        VecCopy_Seq(y, w, n)
        PCApplySymmetric(pci, pcf, pstmp, n, w, y)


# ---------------------------------------------------------------- CG


def KSPSolve_CG(
    n: Int, arow: PI, acol: PI, adata: Ptr, pci: PI, pcf: Ptr, pstmp: Ptr,
    work: Int, cfg: Ptr, cfgi: PI, b: Ptr, x: Ptr
):
    """cg.c KSPSolve_CG.

    The Steihaugh-Toint trust region (`cg->radius`, `norm_d`, `norm_p`, `dMp`),
    the objective test (`cg->obj`, `obj_min`) and the eigenvalue collection
    (`ksp->calc_sings`) are not ported.  Everything else is, including the
    indefinite-preconditioner and indefinite-matrix breaks, the `chknorm`
    bookkeeping that defers some norm and preconditioner applications, and the
    `W = Z` aliasing of the work vectors.
    """
    var normtype = cfgi.unsafe_load(0)
    var pc_side = cfgi.unsafe_load(1)
    var guess_zero = cfgi.unsafe_load(2)
    var chknorm = cfgi.unsafe_load(4)
    var max_it = cfgi.unsafe_load(6)

    var X = x
    var B = b
    var R = _vec(work, 0, n)
    var Z = _vec(work, 1, n)
    var P = _vec(work, 2, n)
    var W = Z  # upstream: W = Z

    var a = 1.0
    var beta = 0.0
    var betaold = 1.0
    var bcoef = 0.0
    var dpi = 0.0
    var dpiold = 0.0
    var dp = 0.0
    var its = 0
    var reason = 0

    if not guess_zero:
        MatMult_SeqAIJ(X, R, arow, acol, adata, n)  # r <- A x
        VecAYPX_Seq(R, -1.0, B, n)  # r <- b - A x
    else:
        VecCopy_Seq(B, R, n)  # r <- b

    if normtype == KSP_NORM_PRECONDITIONED:
        KSP_PCApply(pci, pcf, pstmp, n, R, Z)  # z <- M⁻¹ r
        dp = VecNorm2(Z, n)
    elif normtype == KSP_NORM_UNPRECONDITIONED:
        dp = VecNorm2(R, n)
    elif normtype == KSP_NORM_NATURAL:
        KSP_PCApply(pci, pcf, pstmp, n, R, Z)
        beta = _dot(Z, R, n)
        dp = sqrt(abs(beta))
    else:
        dp = 0.0

    cfg.unsafe_store(9, dp)
    reason = KSPConvergedDefault(0, dp, cfg, cfgi)
    if reason != 0:
        _finish(cfg, cfgi, dp, its, reason)
        return

    if normtype != KSP_NORM_PRECONDITIONED and normtype != KSP_NORM_NATURAL:
        KSP_PCApply(pci, pcf, pstmp, n, R, Z)
    if normtype != KSP_NORM_NATURAL:
        beta = _dot(Z, R, n)

    var i = 0
    while True:
        its = i + 1
        if beta == 0.0:
            reason = KSP_CONVERGED_ATOL
            break
        elif i > 0 and beta * betaold < 0.0:
            reason = KSP_DIVERGED_INDEFINITE_PC
            break
        if not i:
            VecCopy_Seq(Z, P, n)  # p <- z
            bcoef = 0.0
        else:
            bcoef = beta / betaold
            VecAYPX_Seq(P, bcoef, Z, n)  # p <- z + b p
        dpiold = dpi
        MatMult_SeqAIJ(P, W, arow, acol, adata, n)  # w <- A p
        dpi = _dot(P, W, n)  # dpi <- p'w
        betaold = beta

        if dpi == 0.0 or (i > 0 and _sign(dpi) * _sign(dpiold) < 0):
            reason = KSP_DIVERGED_INDEFINITE_MAT
            break
        a = beta / dpi

        VecAXPY_Seq(X, a, P, n)  # x <- x + a p
        VecAXPY_Seq(R, -a, W, n)  # r <- r - a w

        if normtype == KSP_NORM_PRECONDITIONED and chknorm < i + 2:
            KSP_PCApply(pci, pcf, pstmp, n, R, Z)
            dp = VecNorm2(Z, n)
        elif normtype == KSP_NORM_UNPRECONDITIONED and chknorm < i + 2:
            dp = VecNorm2(R, n)
        elif normtype == KSP_NORM_NATURAL:
            KSP_PCApply(pci, pcf, pstmp, n, R, Z)
            beta = _dot(Z, R, n)
            dp = sqrt(abs(beta))
        else:
            dp = 0.0
        cfg.unsafe_store(9 + its, dp)
        reason = KSPConvergedDefault(its, dp, cfg, cfgi)
        if reason != 0:
            break

        if (normtype != KSP_NORM_PRECONDITIONED and normtype != KSP_NORM_NATURAL) or chknorm >= i + 2:
            KSP_PCApply(pci, pcf, pstmp, n, R, Z)
        if normtype != KSP_NORM_NATURAL or chknorm >= i + 2:
            beta = _dot(Z, R, n)
        i += 1
        if i >= max_it:
            break
    if reason == 0:
        reason = KSP_DIVERGED_ITS
    _finish(cfg, cfgi, dp, its, reason)


# ---------------------------------------------------------------- BiCGStab


def KSPSolve_BCGS(
    n: Int, arow: PI, acol: PI, adata: Ptr, pci: PI, pcf: Ptr, pstmp: Ptr,
    work: Int, cfg: Ptr, cfgi: PI, b: Ptr, x: Ptr
):
    """bcgs.c KSPSolve_BCGS: stabilized biconjugate gradients, PETSc 3.25 form.

    PETSc 3.25 has no omega-near-zero break; the only breakdowns are the exact
    `d1 == 0`, `d2 == 0` and `rho == 0` tests below.
    """
    var normtype = cfgi.unsafe_load(0)
    var pc_side = cfgi.unsafe_load(1)
    var guess_zero = cfgi.unsafe_load(2)
    var chknorm = cfgi.unsafe_load(4)
    var max_it = cfgi.unsafe_load(6)

    var X = x
    var B = b
    var R = _vec(work, 0, n)
    var RP = _vec(work, 1, n)
    var V = _vec(work, 2, n)
    var T = _vec(work, 3, n)
    var S = _vec(work, 4, n)
    var P = _vec(work, 5, n)
    var GUESS = _vec(work, 6, n)
    var has_guess = 0

    KSPInitialResidual(
        pci, pcf, pstmp, pc_side, guess_zero, arow, acol, adata, n,
        X, V, T, R, B
    )
    if pc_side == PC_RIGHT and not guess_zero:
        VecCopy_Seq(X, GUESS, n)
        VecSet_Seq(X, 0.0, n)
        has_guess = 1

    var dp = VecNorm2(R, n) if normtype != KSP_NORM_NONE else 0.0
    var its = 0
    cfg.unsafe_store(9, dp)
    var reason = KSPConvergedDefault(0, dp, cfg, cfgi)
    if reason != 0:
        if has_guess:
            VecAXPY_Seq(X, 1.0, GUESS, n)
        _finish(cfg, cfgi, dp, its, reason)
        return

    VecCopy_Seq(R, RP, n)
    var rhoold = 1.0
    var alpha = 1.0
    var omegaold = 1.0
    var omega = 1.0
    var rho = 1.0
    var d1 = 0.0
    var d2 = 0.0
    VecSet_Seq(P, 0.0, n)
    VecSet_Seq(V, 0.0, n)

    var i = 0
    while True:
        rho = _dot(R, RP, n)
        var beta = (rho / rhoold) * (alpha / omegaold)
        VecAXPBYPCZ_Seq(P, 1.0, -omegaold * beta, beta, R, V, n)
        KSP_PCApplyBAorAB(
            pci, pcf, pstmp, pc_side, arow, acol, adata, n, P, V, T
        )  # v <- M⁻¹ A p
        d1 = _dot(V, RP, n)
        if d1 == 0.0:
            reason = KSP_DIVERGED_BREAKDOWN
            break
        alpha = rho / d1
        VecWAXPY_Seq(S, -alpha, V, R, n)  # s = -alpha v + r
        KSP_PCApplyBAorAB(
            pci, pcf, pstmp, pc_side, arow, acol, adata, n, S, T, R
        )  # t <- M⁻¹ A s
        # VecDotNorm2(S, T, &d1, &d2)
        d1 = _dot(S, T, n)
        d2 = _snorm2(S, n)
        if d2 == 0.0:
            d1 = _dot(S, S, n)
            if d1 != 0.0:
                reason = KSP_DIVERGED_BREAKDOWN
                break
            VecAXPY_Seq(X, alpha, P, n)
            its += 1
            dp = 0.0
            reason = KSP_CONVERGED_RTOL
            break
        omega = d1 / d2
        VecAXPBYPCZ_Seq(X, alpha, omega, 1.0, P, S, n)
        VecWAXPY_Seq(R, -omega, T, S, n)
        if normtype != KSP_NORM_NONE and chknorm < i + 2:
            dp = VecNorm2(R, n)
        rhoold = rho
        omegaold = omega
        its += 1
        cfg.unsafe_store(9 + its, dp)
        reason = KSPConvergedDefault(its, dp, cfg, cfgi)
        if reason != 0:
            break
        if rho == 0.0:
            reason = KSP_DIVERGED_BREAKDOWN
            break
        i += 1
        if i >= max_it:
            break
    if reason == 0:
        reason = KSP_DIVERGED_ITS
    KSPUnwindPreconditioner(pci, pcf, pstmp, pc_side, n, X, T)
    if has_guess:
        VecAXPY_Seq(X, 1.0, GUESS, n)
    _finish(cfg, cfgi, dp, its, reason)


# ---------------------------------------------------------------- GMRES


def KSPGMRESModifiedGramSchmidtOrthogonalization(
    vv: Int, hh: Ptr, hes: Ptr, m: Int, it: Int, n: Int
):
    """borthog.c KSPGMRESModifiedGramSchmidtOrthogonalization."""
    var k = 0
    while k <= it:
        var d = _dot(_vv(vv, k, n), _vv(vv, it + 1, n), n)
        # HES(a, b) = hes + b * (max_k + 1) + a
        hes.unsafe_store(it * (m + 1) + k, d)
        VecAXPY_Seq(_vv(vv, it + 1, n), -d, _vv(vv, k, n), n)
        k += 1


def KSPGMRESClassicalGramSchmidtOrthogonalization(
    vv: Int, hh: Ptr, hes: Ptr, lhh: Ptr, m: Int, it: Int, n: Int, cgstype: Int
):
    """borthog2.c KSPGMRESClassicalGramSchmidtOrthogonalization.

    The refinement step re-dots the already projected vector and accumulates
    into `hh`/`hes`; that accumulation is what makes it a refinement rather
    than a replacement.
    """
    var refine = 1 if cgstype == CGS_REFINE_ALWAYS else 0
    var j = 0
    while j <= it:
        # HH(a, b) = hh + b * (max_k + 2) + a
        hh.unsafe_store(it * (m + 2) + j, 0.0)
        hes.unsafe_store(it * (m + 1) + j, 0.0)
        j += 1

    _mdot_vv(vv, it + 1, vv, lhh, n)
    j = 0
    while j <= it:
        lhh.unsafe_store(j, -lhh.unsafe_load(j))
        j += 1
    _maxpy_vv(_vv(vv, it + 1, n), lhh, vv, it + 1, n)
    j = 0
    while j <= it:
        var hv = hh.unsafe_load(it * (m + 2) + j) - lhh.unsafe_load(j)
        hh.unsafe_store(it * (m + 2) + j, hv)
        var sv = hes.unsafe_load(it * (m + 1) + j) - lhh.unsafe_load(j)
        hes.unsafe_store(it * (m + 1) + j, sv)
        j += 1

    if cgstype == CGS_REFINE_IFNEEDED:
        var hnrm = 0.0
        j = 0
        while j <= it:
            hnrm += lhh.unsafe_load(j) * lhh.unsafe_load(j)
            j += 1
        hnrm = sqrt(hnrm)
        if VecNorm2(_vv(vv, it + 1, n), n) < hnrm:
            refine = 1
    if refine:
        _mdot_vv(vv, it + 1, vv, lhh, n)
        j = 0
        while j <= it:
            lhh.unsafe_store(j, -lhh.unsafe_load(j))
            j += 1
        _maxpy_vv(_vv(vv, it + 1, n), lhh, vv, it + 1, n)
        j = 0
        while j <= it:
            var hv = hh.unsafe_load(it * (m + 2) + j) - lhh.unsafe_load(j)
            hh.unsafe_store(it * (m + 2) + j, hv)
            var sv = hes.unsafe_load(it * (m + 1) + j) - lhh.unsafe_load(j)
            hes.unsafe_store(it * (m + 1) + j, sv)
            j += 1


def KSPGMRESUpdateHessenberg(
    hh: Ptr, cc: Ptr, ss: Ptr, rs: Ptr, it: Int, m: Int, hapend: Int
) -> Tuple[Float64, Int]:
    """gmres.c KSPGMRESUpdateHessenberg.  Returns (residual norm, reason).

    Applies the previously computed plane rotations down the new column, then
    computes the new rotation and folds it into both the right-hand side and
    the Hessenberg column; the residual is the new right-hand side entry.
    """
    var j = 1
    while j <= it:
        # `hh` walks HH(0, it), HH(1, it), ... down the new column
        var idx = it * (m + 2) + (j - 1)
        var tt = hh.unsafe_load(idx)
        hh.unsafe_store(
            idx,
            cc.unsafe_load(j - 1) * tt
            + ss.unsafe_load(j - 1) * hh.unsafe_load(idx + 1),
        )
        hh.unsafe_store(
            idx + 1,
            cc.unsafe_load(j - 1) * hh.unsafe_load(idx + 1)
            - ss.unsafe_load(j - 1) * tt,
        )
        j += 1
    if not hapend:
        var base = it * (m + 2) + it
        var a = hh.unsafe_load(base)
        var b = hh.unsafe_load(base + 1)
        var tt = sqrt(a * a + b * b)
        if tt == 0.0:
            return (0.0, KSP_DIVERGED_NULL)
        cc.unsafe_store(it, a / tt)
        ss.unsafe_store(it, b / tt)
        rs.unsafe_store(it + 1, -ss.unsafe_load(it) * rs.unsafe_load(it))
        rs.unsafe_store(it, cc.unsafe_load(it) * rs.unsafe_load(it))
        hh.unsafe_store(
            base, cc.unsafe_load(it) * a + ss.unsafe_load(it) * b
        )
        return (abs(rs.unsafe_load(it + 1)), 0)
    return (0.0, 0)


def KSPGMRESBuildSoln(
    rs: Ptr, hh: Ptr, vv: Int, temp: Ptr, wtmp: Ptr, vs: Ptr, vdest: Ptr, m: Int,
    it: Int, n: Int, pci: PI, pcf: Ptr, pstmp: Ptr, pc_side: Int
) -> Int:
    """gmres.c KSPGMRESBuildSoln.  Returns the reason set on a zero pivot.

    `nrs` aliases `rs`, exactly as upstream passes `GRS(0)` for the work area.
    """
    if it < 0:
        VecCopy_Seq(vs, vdest, n)
        return 0
    if hh.unsafe_load(it * (m + 2) + it) != 0.0:
        rs.unsafe_store(it, rs.unsafe_load(it) / hh.unsafe_load(it * (m + 2) + it))
    else:
        return KSP_DIVERGED_BREAKDOWN
    var ii = 1
    while ii <= it:
        var k = it - ii
        var tt = rs.unsafe_load(k)
        var j = k + 1
        while j <= it:
            # R(k, j) is HH(k, j), and HH(a, b) = hh + b * (max_k + 2) + a
            tt -= hh.unsafe_load(j * (m + 2) + k) * rs.unsafe_load(j)
            j += 1
        if hh.unsafe_load(k * (m + 2) + k) == 0.0:
            return KSP_DIVERGED_BREAKDOWN
        rs.unsafe_store(k, tt / hh.unsafe_load(k * (m + 2) + k))
        ii += 1
    # VecMAXPBY(VEC_TEMP, it + 1, nrs, 0, &VEC_VV(0))
    _maxpy_vv_n(temp, rs, vv, it + 1, n)
    KSPUnwindPreconditioner(pci, pcf, pstmp, pc_side, n, temp, wtmp)
    VecAXPY_Seq(vdest, 1.0, temp, n)
    return 0


def KSPSolve_GMRES(
    n: Int, arow: PI, acol: PI, adata: Ptr, pci: PI, pcf: Ptr, pstmp: Ptr,
    work: Int, vv: Int, hs: Int, cfg: Ptr, cfgi: PI, b: Ptr, x: Ptr
):
    """gmres.c KSPSolve_GMRES: the restart driver around KSPGMRESCycle.

    The singular-value and Ritz-pair collection paths, the Krylov-vector
    monitor, and the incremental `KSPGMRESGetNewVectors` allocation (the basis
    is preallocated here instead) are not ported.
    """
    var normtype = cfgi.unsafe_load(0)
    var pc_side = cfgi.unsafe_load(1)
    var guess_zero = cfgi.unsafe_load(2)
    var max_it = cfgi.unsafe_load(6)
    var m = cfgi.unsafe_load(7)
    var orthog = cfgi.unsafe_load(8)
    var cgstype = cfgi.unsafe_load(9)
    var haptol = cfg.unsafe_load(3)
    var breakdowntol = cfg.unsafe_load(4)

    var HH = hs_off(hs, m, 0)
    var HES = hs_off(hs, m, 1)
    var CC = hs_off(hs, m, 2)
    var SS = hs_off(hs, m, 3)
    var RS = hs_off(hs, m, 4)
    var ORTHOG = hs_off(hs, m, 5)
    var TEMP = _vec(work, 0, n)
    var TEMP_MATOP = _vec(work, 1, n)

    var its = 0
    var itcount = 0
    var reason = 0
    var res = 0.0
    var prev_rnorm = -1.0

    while reason == 0:
        KSPInitialResidual(
            pci, pcf, pstmp, pc_side, guess_zero, arow, acol, adata, n,
            x, TEMP, TEMP_MATOP, _vv(vv, 0, n), b
        )
        # ---- KSPGMRESCycle
        var it = 0
        res = _normalize(_vv(vv, 0, n), n)
        if prev_rnorm > 0.0 and abs(res - prev_rnorm) > breakdowntol * cfg.unsafe_load(8):
            reason = KSP_DIVERGED_BREAKDOWN
            _finish(cfg, cfgi, res, its, reason)
            return
        RS.unsafe_store(0, res)
        cfg.unsafe_store(8, res)
        cfg.unsafe_store(9 + its, res)
        if not res:
            reason = KSP_CONVERGED_ATOL
        else:
            reason = KSPConvergedDefault(its, res, cfg, cfgi)
        var hapend = 0
        while reason == 0 and it < m and its < max_it:
            KSP_PCApplyBAorAB(
                pci, pcf, pstmp, pc_side, arow, acol, adata, n,
                _vv(vv, it, n), _vv(vv, it + 1, n), TEMP_MATOP
            )
            if orthog == ORTHOG_MGS:
                KSPGMRESModifiedGramSchmidtOrthogonalization(vv, HH, HES, m, it, n)
            else:
                KSPGMRESClassicalGramSchmidtOrthogonalization(
                    vv, HH, HES, ORTHOG, m, it, n, cgstype
                )
            var tt = _normalize(_vv(vv, it + 1, n), n)
            # HH(a, b) = hh + b * (max_k + 2) + a, so the subdiagonal entry
            # HH(it + 1, it) is row it + 1 of column it
            HH.unsafe_store(it * (m + 2) + it + 1, tt)
            HES.unsafe_store(it * (m + 1) + it + 1, tt)
            var hapbnd = abs(tt / RS.unsafe_load(it))
            hapbnd = haptol if hapbnd > haptol else hapbnd
            hapend = 1 if tt < hapbnd else 0
            var r2 = KSPGMRESUpdateHessenberg(HH, CC, SS, RS, it, m, hapend)
            res = r2[0]
            reason = r2[1]
            it += 1
            its += 1
            cfg.unsafe_store(9 + its, res)
            if reason != 0:
                break
            reason = KSPConvergedDefault(its, res, cfg, cfgi)
            if hapend:
                if normtype == KSP_NORM_NONE:
                    reason = KSP_CONVERGED_HAPPY_BREAKDOWN
                elif reason == 0:
                    reason = KSP_DIVERGED_BREAKDOWN
                    break
        itcount += it
        var breason = KSPGMRESBuildSoln(
            RS, HH, vv, TEMP, TEMP_MATOP, x, x, m, it - 1, n, pci, pcf, pstmp,
            pc_side
        )
        if reason == 0 and breason != 0:
            reason = breason
        if itcount >= max_it:
            if reason == 0:
                reason = KSP_DIVERGED_ITS
            break
        prev_rnorm = res
        guess_zero = 0
    _finish(cfg, cfgi, res, its, reason)


# ---------------------------------------------------------------- FGMRES


def KSPSolve_FGMRES(
    n: Int, arow: PI, acol: PI, adata: Ptr, pci: PI, pcf: Ptr, pstmp: Ptr,
    work: Int, vv: Int, prevec: Int, hs: Int, cfg: Ptr, cfgi: PI, b: Ptr, x: Ptr
):
    """fgmres.c KSPSolve_FGMRES: restarted GMRES with a variable preconditioner.

    The `modifypc` callback (`KSPFlexibleSetModifyPC`) is not ported, so the
    preconditioner is fixed across the solve; the preconditioned directions are
    still stored in `prevec` and the solution is assembled from them, which is
    what makes the method flexible.
    """
    var pc_side = cfgi.unsafe_load(1)
    var guess_zero = cfgi.unsafe_load(2)
    var max_it = cfgi.unsafe_load(6)
    var m = cfgi.unsafe_load(7)
    var orthog = cfgi.unsafe_load(8)
    var cgstype = cfgi.unsafe_load(9)
    var haptol = cfg.unsafe_load(3)

    var HH = hs_off(hs, m, 0)
    var HES = hs_off(hs, m, 1)
    var CC = hs_off(hs, m, 2)
    var SS = hs_off(hs, m, 3)
    var RS = hs_off(hs, m, 4)
    var ORTHOG = hs_off(hs, m, 5)
    var TEMP = _vec(work, 0, n)
    var TEMP_MATOP = _vec(work, 1, n)

    var its = 0
    var reason = 0
    var res_norm = 0.0

    if not guess_zero:
        _fgmres_residual(arow, acol, adata, n, x, TEMP, vv, b)
    else:
        VecCopy_Seq(b, _vv(vv, 0, n), n)

    while True:
        # ---- KSPFGMRESCycle
        var loc_it = 0
        res_norm = VecNorm2(_vv(vv, 0, n), n)
        RS.unsafe_store(0, res_norm)
        cfg.unsafe_store(9 + its, res_norm)
        reason = KSPConvergedDefault(its, res_norm, cfg, cfgi)
        if reason == 0:
            VecScale_Seq(_vv(vv, 0, n), 1.0 / res_norm, n)
            var hapend = 0
            while reason == 0 and loc_it < m and its < max_it:
                KSP_PCApply(
                    pci, pcf, pstmp, n, _vv(vv, loc_it, n), _vv(prevec, loc_it, n)
                )
                MatMult_SeqAIJ(
                    _vv(prevec, loc_it, n), _vv(vv, loc_it + 1, n),
                    arow, acol, adata, n
                )
                if orthog == ORTHOG_MGS:
                    KSPGMRESModifiedGramSchmidtOrthogonalization(
                        vv, HH, HES, m, loc_it, n
                    )
                else:
                    KSPGMRESClassicalGramSchmidtOrthogonalization(
                        vv, HH, HES, ORTHOG, m, loc_it, n, cgstype
                    )
                var tt = VecNorm2(_vv(vv, loc_it + 1, n), n)
                HH.unsafe_store(loc_it * (m + 2) + loc_it + 1, tt)
                HES.unsafe_store(loc_it * (m + 1) + loc_it + 1, tt)
                var hapbnd = abs(tt / RS.unsafe_load(loc_it))
                hapbnd = haptol if hapbnd > haptol else hapbnd
                if tt > hapbnd:
                    VecScale_Seq(_vv(vv, loc_it + 1, n), 1.0 / tt, n)
                else:
                    VecSet_Seq(TEMP, 0.0, n)
                    hapend = 1
                var r2 = KSPGMRESUpdateHessenberg(HH, CC, SS, RS, loc_it, m, hapend)
                res_norm = r2[0]
                reason = r2[1]
                if reason != 0:
                    break
                loc_it += 1
                its += 1
                cfg.unsafe_store(9 + its, res_norm)
                reason = KSPConvergedDefault(its, res_norm, cfg, cfgi)
                if hapend and reason == 0:
                    reason = KSP_DIVERGED_BREAKDOWN
                    break
        _fgmres_buildsoln(RS, HH, prevec, TEMP, m, loc_it - 1, n, x)
        if its >= max_it:
            if reason == 0:
                reason = KSP_DIVERGED_ITS
            break
        if reason != 0:
            break
        _fgmres_residual(arow, acol, adata, n, x, TEMP, vv, b)
    _finish(cfg, cfgi, res_norm, its, reason)


def _fgmres_residual(
    arow: PI, acol: PI, adata: Ptr, n: Int, x: Ptr, temp: Ptr, vv: Int,
    b: Ptr
):
    """fgmres.c KSPFGMRESResidual: VEC_VV(0) = b - A x, unpreconditioned."""
    MatMult_SeqAIJ(x, temp, arow, acol, adata, n)
    VecWAXPY_Seq(_vv(vv, 0, n), -1.0, temp, b, n)


def _fgmres_buildsoln(
    rs: Ptr, hh: Ptr, prevec: Int, temp: Ptr, m: Int, it: Int, n: Int, vdest: Ptr
):
    """fgmres.c KSPFGMRESBuildSoln: back-substitute against HH, then assemble
    the correction from the *preconditioned* directions."""
    if it < 0:
        return
    if hh.unsafe_load(it * (m + 2) + it) != 0.0:
        rs.unsafe_store(it, rs.unsafe_load(it) / hh.unsafe_load(it * (m + 2) + it))
    else:
        rs.unsafe_store(it, 0.0)
    var ii = 1
    while ii <= it:
        var k = it - ii
        var tt = rs.unsafe_load(k)
        var j = k + 1
        while j <= it:
            # R(k, j) is HH(k, j), and HH(a, b) = hh + b * (max_k + 2) + a
            tt -= hh.unsafe_load(j * (m + 2) + k) * rs.unsafe_load(j)
            j += 1
        rs.unsafe_store(k, tt / hh.unsafe_load(k * (m + 2) + k))
        ii += 1
    _maxpy_vv_n(temp, rs, prevec, it + 1, n)
    VecAXPY_Seq(vdest, 1.0, temp, n)


# ---------------------------------------------------------------- helpers


def _vec(base: Int, k: Int, n: Int) -> Ptr:
    """Vector k of a flat Float64 work arena."""
    return Ptr(unsafe_from_address=base + k * n * 8)


def _vv(base: Int, k: Int, n: Int) -> Ptr:
    """Vector k of a flat (m + 1) x n basis block."""
    return Ptr(unsafe_from_address=base + k * n * 8)


def hs_off(hs: Int, m: Int, which: Int) -> Ptr:
    """Offset into the GMRES scalar block; see the module docstring."""
    var hhn = (m + 2) * (m + 1)
    var hesn = (m + 1) * (m + 1)
    var ccn = m + 1
    var rsn = m + 2
    if which == 0:
        return Ptr(unsafe_from_address=hs)
    if which == 1:
        return Ptr(unsafe_from_address=hs + hhn * 8)
    if which == 2:
        return Ptr(unsafe_from_address=hs + (hhn + hesn) * 8)
    if which == 3:
        return Ptr(unsafe_from_address=hs + (hhn + hesn + ccn) * 8)
    if which == 4:
        return Ptr(unsafe_from_address=hs + (hhn + hesn + 2 * ccn) * 8)
    return Ptr(unsafe_from_address=hs + (hhn + hesn + 2 * ccn + rsn) * 8)


def _sign(x: Float64) -> Int:
    """PetscSignReal."""
    if x < 0.0:
        return -1
    if x == 0.0:
        return 0
    return 1


def _dot(x: Ptr, y: Ptr, n: Int) -> Float64:
    var s = 0.0
    var i = 0
    while i < n:
        s += x.unsafe_load(i) * y.unsafe_load(i)
        i += 1
    return s


def _snorm2(x: Ptr, n: Int) -> Float64:
    """The `rval` half of VecDotNorm2: the squared norm of x."""
    var r = 0.0
    var i = 0
    while i < n:
        r += x.unsafe_load(i) * x.unsafe_load(i)
        i += 1
    return r


def _mdot_vv(base: Int, k: Int, other: Int, res: Ptr, n: Int):
    """VecMDot(VV(k), k + 1, &VV(0), out) over a flat basis block.

    `VecMDot` with `nv = k + 1` produces entries 0..k, so the loop is exclusive.
    """
    var j = 0
    while j < k:
        res.unsafe_store(j, _dot(_vv(base, k, n), _vv(other, j, n), n))
        j += 1


def _maxpy_vv(y: Ptr, alpha: Ptr, base: Int, nvec: Int, n: Int):
    """VecMAXPY(y, nvec, alpha, &VV(0)) over a flat basis block."""
    var k = 0
    while k < nvec:
        var yk = _vv(base, k, n)
        var a = alpha.unsafe_load(k)
        var i = 0
        while i < n:
            y.unsafe_store(i, y.unsafe_load(i) + a * yk.unsafe_load(i))
            i += 1
        k += 1


def _maxpy_vv_n(y: Ptr, alpha: Ptr, base: Int, nvec: Int, n: Int):
    """VecMAXPBY(y, nvec, alpha, 0, &VV(0)).

    beta = 0 replaces `y` with the sum, so the accumulation still runs over
    every term; it just does not add the old `y` in.
    """
    var i = 0
    while i < n:
        y.unsafe_store(i, 0.0)
        i += 1
    var k = 0
    while k < nvec:
        var yk = _vv(base, k, n)
        var a = alpha.unsafe_load(k)
        i = 0
        while i < n:
            y.unsafe_store(i, y.unsafe_load(i) + a * yk.unsafe_load(i))
            i += 1
        k += 1


def _normalize(x: Ptr, n: Int) -> Float64:
    """VecNormalize, returning the norm."""
    var z = VecNorm2(x, n)
    if z > 0.0:
        var i = 0
        while i < n:
            x.unsafe_store(i, x.unsafe_load(i) / z)
            i += 1
    return z


def _finish(cfg: Ptr, cfgi: PI, rnorm: Float64, its: Int, reason: Int):
    cfg.unsafe_store(7, rnorm)
    cfgi.unsafe_store(10, its)
    cfgi.unsafe_store(11, reason)

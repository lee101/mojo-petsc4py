"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address: `@export` rejects parametric functions
and a pointer with an inferred origin is parametric.  Callers own all memory,
including scratch, so nothing here allocates and nothing here can leak.  These
wrappers are deliberately thin; the algorithms live in `ported.mojo` and
`ksp.mojo` under their PETSc names.


Part of mojo-petsc4py (MIT); the kernels it exports are transcribed from
PETSc 3.25.5 / petsc4py 3.25.5 (BSD-2-Clause), whose notices are in NOTICE.
"""

from factor import *
from ksp import *
from ported import *


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def pi(addr: Int) -> PI:
    return PI(unsafe_from_address=addr)


def pc(addr: Int) -> PI:
    return PI(unsafe_from_address=addr)


# ---------------------------------------------------------------- Mat
@export("mp_mat_mult")
def mp_mat_mult(x: Int, y: Int, irow: Int, icol: Int, data: Int, nrow: Int) abi("C"):
    MatMult_SeqAIJ(p(x), p(y), pc(irow), pc(icol), p(data), nrow)


@export("mp_mat_mult_add")
def mp_mat_mult_add(
    x: Int, y: Int, irow: Int, icol: Int, data: Int, nrow: Int
) abi("C"):
    MatMultAdd_SeqAIJ(p(x), p(y), pc(irow), pc(icol), p(data), nrow)


@export("mp_mat_mult_transpose")
def mp_mat_mult_transpose(
    x: Int, y: Int, irow: Int, icol: Int, data: Int, nrow: Int, ncols: Int
) abi("C"):
    MatMultTranspose_SeqAIJ(p(x), p(y), pc(irow), pc(icol), p(data), nrow, ncols)


@export("mp_mat_get_diagonal")
def mp_mat_get_diagonal(
    diag: Int, irow: Int, icol: Int, data: Int, nrow: Int
) abi("C"):
    MatGetDiagonal_SeqAIJ(p(diag), pc(irow), pc(icol), p(data), nrow)


@export("mp_mat_scale")
def mp_mat_scale(data: Int, nnz: Int, alpha: Float64) abi("C"):
    MatScale_SeqAIJ(p(data), nnz, alpha)


@export("mp_mat_zero_entries")
def mp_mat_zero_entries(data: Int, nnz: Int) abi("C"):
    MatZeroEntries_SeqAIJ(p(data), nnz)


# ---------------------------------------------------------------- Vec
@export("mp_vec_pointwise_mult")
def mp_vec_pointwise_mult(w: Int, x: Int, y: Int, n: Int) abi("C"):
    VecPointwiseMult_Seq(p(w), p(x), p(y), n)


@export("mp_vec_pointwise_divide")
def mp_vec_pointwise_divide(w: Int, x: Int, y: Int, n: Int) abi("C"):
    VecPointwiseDivide_Seq(p(w), p(x), p(y), n)


@export("mp_vec_pointwise_min")
def mp_vec_pointwise_min(w: Int, x: Int, y: Int, n: Int) abi("C"):
    VecPointwiseMin_Seq(p(w), p(x), p(y), n)


@export("mp_vec_pointwise_max")
def mp_vec_pointwise_max(w: Int, x: Int, y: Int, n: Int) abi("C"):
    VecPointwiseMax_Seq(p(w), p(x), p(y), n)


@export("mp_vec_pointwise_max_abs")
def mp_vec_pointwise_max_abs(w: Int, x: Int, y: Int, n: Int) abi("C"):
    VecPointwiseMaxAbs_Seq(p(w), p(x), p(y), n)


@export("mp_vec_norm")
def mp_vec_norm(x: Int, ntype: Int, z: Int, n: Int) abi("C"):
    VecNorm_Seq(p(x), ntype, p(z), n)


@export("mp_vec_dot")
def mp_vec_dot(x: Int, y: Int, val: Int, n: Int) abi("C"):
    VecDot_Seq(p(x), p(y), p(val), n)


@export("mp_vec_dot_norm2")
def mp_vec_dot_norm2(x: Int, y: Int, sval: Int, rval: Int, n: Int) abi("C"):
    VecDotNorm2_Seq(p(x), p(y), p(sval), p(rval), n)


@export("mp_vec_sum")
def mp_vec_sum(x: Int, val: Int, n: Int) abi("C"):
    VecSum_Seq(p(x), p(val), n)


@export("mp_vec_mean")
def mp_vec_mean(x: Int, val: Int, n: Int) abi("C"):
    VecMean_Seq(p(x), p(val), n)


@export("mp_vec_max")
def mp_vec_max(x: Int, idx: Int, z: Int, n: Int) abi("C"):
    VecMax_Seq(p(x), pi(idx), p(z), n)


@export("mp_vec_min")
def mp_vec_min(x: Int, idx: Int, z: Int, n: Int) abi("C"):
    VecMin_Seq(p(x), pi(idx), p(z), n)


@export("mp_vec_normalize")
def mp_vec_normalize(x: Int, v: Int, n: Int) abi("C"):
    VecNormalize_Seq(p(x), p(v), n)


@export("mp_vec_reciprocal")
def mp_vec_reciprocal(x: Int, n: Int) abi("C"):
    VecReciprocal_Seq(p(x), n)


@export("mp_vec_exp")
def mp_vec_exp(x: Int, n: Int) abi("C"):
    VecExp_Seq(p(x), n)


@export("mp_vec_log")
def mp_vec_log(x: Int, n: Int) abi("C"):
    VecLog_Seq(p(x), n)


@export("mp_vec_sqrtabs")
def mp_vec_sqrtabs(x: Int, n: Int) abi("C"):
    VecSqrtAbs_Seq(p(x), n)


@export("mp_vec_chop")
def mp_vec_chop(x: Int, n: Int, tol: Float64) abi("C"):
    VecChop_Seq(p(x), n, tol)


@export("mp_vec_abs")
def mp_vec_abs(x: Int, n: Int) abi("C"):
    VecAbs_Seq(p(x), n)


@export("mp_vec_conjugate")
def mp_vec_conjugate(x: Int, n: Int) abi("C"):
    VecConjugate_Seq(p(x), n)


@export("mp_vec_copy")
def mp_vec_copy(x: Int, y: Int, n: Int) abi("C"):
    VecCopy_Seq(p(x), p(y), n)


@export("mp_vec_swap")
def mp_vec_swap(x: Int, y: Int, n: Int) abi("C"):
    VecSwap_Seq(p(x), p(y), n)


@export("mp_vec_set")
def mp_vec_set(x: Int, alpha: Float64, n: Int) abi("C"):
    VecSet_Seq(p(x), alpha, n)


@export("mp_vec_zero_entries")
def mp_vec_zero_entries(x: Int, n: Int) abi("C"):
    VecZeroEntries_Seq(p(x), n)


@export("mp_vec_scale")
def mp_vec_scale(x: Int, alpha: Float64, n: Int) abi("C"):
    VecScale_Seq(p(x), alpha, n)


@export("mp_vec_shift")
def mp_vec_shift(x: Int, alpha: Float64, n: Int) abi("C"):
    VecShift_Seq(p(x), alpha, n)


@export("mp_vec_axpy")
def mp_vec_axpy(y: Int, alpha: Float64, x: Int, n: Int) abi("C"):
    VecAXPY_Seq(p(y), alpha, p(x), n)


@export("mp_vec_aypx")
def mp_vec_aypx(y: Int, alpha: Float64, x: Int, n: Int) abi("C"):
    VecAYPX_Seq(p(y), alpha, p(x), n)


@export("mp_vec_axpby")
def mp_vec_axpby(y: Int, alpha: Float64, beta: Float64, x: Int, n: Int) abi("C"):
    VecAXPBY_Seq(p(y), alpha, beta, p(x), n)


@export("mp_vec_axpbypcz")
def mp_vec_axpbypcz(
    z: Int, alpha: Float64, beta: Float64, gamma: Float64, x: Int, y: Int, n: Int
) abi("C"):
    VecAXPBYPCZ_Seq(p(z), alpha, beta, gamma, p(x), p(y), n)


@export("mp_vec_waxpy")
def mp_vec_waxpy(w: Int, alpha: Float64, x: Int, y: Int, n: Int) abi("C"):
    VecWAXPY_Seq(p(w), alpha, p(x), p(y), n)


@export("mp_vec_tdot")
def mp_vec_tdot(x: Int, y: Int, val: Int, n: Int) abi("C"):
    VecTDot_Seq(p(x), p(y), p(val), n)


@export("mp_vec_maxpy")
def mp_vec_maxpy(y: Int, nv: Int, alpha: Int, x: Int, n: Int) abi("C"):
    VecMAXPY_Seq(p(y), nv, p(alpha), pc(x), n)


@export("mp_vec_max_pointwise_divide")
def mp_vec_max_pointwise_divide(x: Int, y: Int, rval: Int, n: Int) abi("C"):
    VecMaxPointwiseDivide_Seq(p(x), p(y), p(rval), n)


# ---------------------------------------------------------------- PC
@export("mp_mat_get_diagonal_markers")
def mp_mat_get_diagonal_markers(
    irow: Int, icol: Int, diag: Int, nrow: Int
) abi("C") -> Int:
    return MatGetDiagonalMarkers_SeqAIJ(pc(irow), pc(icol), pi(diag), nrow)


@export("mp_pc_jacobi_setup")
def mp_pc_jacobi_setup(
    diag: Int, jtype: Int, useabs: Int, fixdiag: Int, scale: Float64,
    irow: Int, icol: Int, data: Int, nrow: Int, didx: Int
) abi("C"):
    PCSetUp_Jacobi(
        p(diag), jtype, useabs, fixdiag, scale, pc(irow), pc(icol), p(data),
        nrow, p(didx)
    )


@export("mp_pc_jacobi_setup_symmetric")
def mp_pc_jacobi_setup_symmetric(
    diagsqrt: Int, jtype: Int, irow: Int, icol: Int, data: Int, nrow: Int
) abi("C"):
    PCSetUp_Jacobi_Symmetric(p(diagsqrt), jtype, pc(irow), pc(icol), p(data), nrow)


@export("mp_pc_jacobi_apply")
def mp_pc_jacobi_apply(y: Int, x: Int, diag: Int, n: Int) abi("C"):
    PCApply_Jacobi(p(y), p(x), p(diag), n)


@export("mp_pc_sor_apply")
def mp_pc_sor_apply(
    x: Int, y: Int, irow: Int, icol: Int, data: Int, dmark: Int, idiag: Int,
    mdiag: Int, t: Int, omega: Float64, fshift: Float64, flag: Int, nrow: Int
) abi("C"):
    PCApply_SOR(
        p(x), p(y), pc(irow), pc(icol), p(data), pi(dmark), p(idiag),
        p(mdiag), p(t), omega, fshift, 1, 1, flag, nrow
    )


@export("mp_pc_ilu_apply")
def mp_pc_ilu_apply(
    b: Int, x: Int, tmp: Int, bi: Int, bj: Int, ba: Int, bdiag: Int, r: Int,
    c: Int, nrow: Int
) abi("C"):
    ILUApplyFactor_SeqAIJ(
        p(b), p(x), p(tmp), pi(bi), pc(bj), p(ba), pi(bdiag), pc(r), pc(c), nrow
    )


@export("mp_pc_icc_apply")
def mp_pc_icc_apply(
    b: Int, x: Int, ui: Int, uj: Int, ua: Int, udiag: Int, c: Int,
    scratch: Int, nrow: Int
) abi("C") -> Int:
    return ICCApplyFactor_SeqAIJ(
        p(b), p(x), pi(ui), pi(uj), p(ua), pi(udiag), pi(c), p(scratch), nrow
    )


@export("mp_pc_ilu_numeric")
def mp_pc_ilu_numeric(
    ba: Int, rtmp: Int, irow: Int, icol: Int, data: Int, bi: Int, bj: Int,
    bdiag: Int, r: Int, ic: Int, nrow: Int
) abi("C") -> Int:
    return MatILUFactorNumeric_SeqAIJ(
        p(ba), p(rtmp), pc(irow), pc(icol), p(data), pi(bi), pc(bj), pi(bdiag),
        pc(r), pc(ic), nrow
    )


@export("mp_pc_icc_numeric")
def mp_pc_icc_numeric(
    ua: Int, rtmp: Int, il: Int, c2r: Int, irow: Int, icol: Int, data: Int,
    ui: Int, uj: Int, udiag: Int, r: Int, riip: Int, nrow: Int
) abi("C"):
    MatICCFactorNumeric_SeqAIJ(
        p(ua), p(rtmp), pi(il), pi(c2r), pc(irow), pc(icol), p(data), pi(ui),
        pi(uj), pi(udiag), pc(r), pc(riip), nrow
    )


@export("mp_ilu_symbolic_ilu0")
def mp_ilu_symbolic_ilu0(
    bi: Int, bj: Int, bdiag: Int, adiag: Int, irow: Int, icol: Int, nrow: Int
) abi("C") -> Int:
    return MatILUFactorSymbolic_SeqAIJ_ilu0(
        pi(bi), pi(bj), pi(bdiag), pi(adiag), pc(irow), pc(icol), nrow
    )


@export("mp_ilu_symbolic")
def mp_ilu_symbolic(
    bi: Int, bj: Int, bjlvl: Int, bdiag: Int, irow: Int, icol: Int, r: Int,
    ic: Int, levels: Int, cap: Int, scratch: Int, nrow: Int
) abi("C") -> Int:
    return MatILUFactorSymbolic_SeqAIJ(
        pi(bi), pi(bj), pi(bjlvl), pi(bdiag), pi(irow), pi(icol), pi(r), pi(ic),
        levels, cap, pi(scratch), nrow
    )


@export("mp_ilu_contiguous_lu")
def mp_ilu_contiguous_lu(
    bjsrc: Int, bi: Int, bdiag: Int, bj: Int, n: Int
) abi("C"):
    MatILUFreeSpaceContiguousLU(pi(bjsrc), pi(bi), pi(bdiag), pi(bj), n)


@export("mp_icc_symbolic")
def mp_icc_symbolic(
    ui: Int, uj: Int, ujlvl: Int, udiag: Int, irow: Int, icol: Int, r: Int,
    riip: Int, levels: Int, cap: Int, scratch: Int, nrow: Int, scratchlen: Int
) abi("C") -> Int:
    return MatICCFactorSymbolic_SeqAIJ(
        pi(ui), pi(uj), pi(ujlvl), pi(udiag), pi(irow), pi(icol), pi(r),
        pi(riip), levels, cap, pi(scratch), nrow, scratchlen
    )


@export("mp_icc_contiguous_cholesky")
def mp_icc_contiguous_cholesky(
    ujsrc: Int, ui: Int, udiag: Int, bi: Int, bdiag: Int, bj: Int, n: Int
) abi("C"):
    MatICCFreeSpaceContiguousCholesky(
        pi(ujsrc), pi(ui), pi(udiag), pi(bi), pi(bdiag), pi(bj), n
    )


@export("mp_ordering_rcm")
def mp_ordering_rcm(
    perm: Int, level: Int, visited: Int, queue: Int, irow: Int, icol: Int,
    nrow: Int
) abi("C"):
    MatGetOrdering_RCM(
        pi(perm), pi(level), pi(visited), pi(queue), pi(irow), pi(icol), nrow
    )


# ---------------------------------------------------------------- KSP
@export("mp_ksp_solve_cg")
def mp_ksp_solve_cg(
    n: Int, irow: Int, icol: Int, data: Int, pci: Int, pcf: Int, pstmp: Int,
    work: Int, cfg: Int, cfgi: Int, b: Int, x: Int
) abi("C"):
    KSPSolve_CG(
        n, pc(irow), pc(icol), p(data), pi(pci), p(pcf), p(pstmp), work,
        p(cfg), pi(cfgi), p(b), p(x)
    )


@export("mp_ksp_solve_bcgs")
def mp_ksp_solve_bcgs(
    n: Int, irow: Int, icol: Int, data: Int, pci: Int, pcf: Int, pstmp: Int,
    work: Int, cfg: Int, cfgi: Int, b: Int, x: Int
) abi("C"):
    KSPSolve_BCGS(
        n, pc(irow), pc(icol), p(data), pi(pci), p(pcf), p(pstmp), work,
        p(cfg), pi(cfgi), p(b), p(x)
    )


@export("mp_ksp_solve_gmres")
def mp_ksp_solve_gmres(
    n: Int, irow: Int, icol: Int, data: Int, pci: Int, pcf: Int, pstmp: Int,
    work: Int, vv: Int, hs: Int, cfg: Int, cfgi: Int, b: Int, x: Int
) abi("C"):
    KSPSolve_GMRES(
        n, pc(irow), pc(icol), p(data), pi(pci), p(pcf), p(pstmp), work, vv,
        hs, p(cfg), pi(cfgi), p(b), p(x)
    )


@export("mp_ksp_solve_fgmres")
def mp_ksp_solve_fgmres(
    n: Int, irow: Int, icol: Int, data: Int, pci: Int, pcf: Int, pstmp: Int,
    work: Int, vv: Int, prevec: Int, hs: Int, cfg: Int, cfgi: Int, b: Int, x: Int
) abi("C"):
    KSPSolve_FGMRES(
        n, pc(irow), pc(icol), p(data), pi(pci), p(pcf), p(pstmp), work, vv,
        prevec, hs, p(cfg), pi(cfgi), p(b), p(x)
    )

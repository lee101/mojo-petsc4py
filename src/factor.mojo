"""Symbolic incomplete factorisation: the level-of-fill sparsity patterns.

`aijfact.c MatILUFactorSymbolic_SeqAIJ` and `MatICCFactorSymbolic_SeqAIJ` build
the pattern with the sorted linked list and per-entry fill levels that
`ported.mojo` implements as `IncompleteLL*`.  The pattern is the same either
way; only the merge rule differs (`PetscILULLAddSorted` versus
`PetscICCLLAddSorted`).

Two divergences, both in allocation only:

* PETSc grows its own free-space blocks as it goes, whereas here the pattern
  is written into a caller-allocated buffer whose capacity is passed in.  When
  a row would not fit, the function returns the negated number of entries
  needed, the caller retries with a larger buffer, and the pattern produced is
  identical.
* The `PetscMalloc` scratch the list machinery needs is one caller-allocated
  block of `5 * nrow + 3` ints (lnk, its levels, the bit set, and, for IC, the
  pivot chain, the first-nonzero map and the column staging buffer) instead of
  five separate allocations.  PETSc's `PetscBT` bit array is kept here as a
  byte array, which is observationally identical.
"""

from ported import *


def MatILUFactorSymbolic_SeqAIJ_ilu0(
    bi: PI, bj: PI, bdiag: PI, adiag: PI, arow: PI, acol: PI, nrow: Int
) -> Int:
    """aijfact.c MatILUFactorSymbolic_SeqAIJ_ilu0.

    ILU(0) with the natural ordering: the L part copies the strictly lower
    triangle, the U part the strictly upper triangle plus the diagonal, with
    the U rows stored in reverse order.  Returns the factor size.
    """
    var k = 0
    bi.unsafe_store(0, 0)
    var i = 0
    while i < nrow:
        var nz = adiag.unsafe_load(i) - arow.unsafe_load(i)
        bi.unsafe_store(i + 1, bi.unsafe_load(i) + nz)
        var src = arow.unsafe_load(i)
        var j = 0
        while j < nz:
            bj.unsafe_store(k, acol.unsafe_load(src + j))
            k += 1
            j += 1
        i += 1
    bdiag.unsafe_store(nrow, bi.unsafe_load(nrow) - 1)
    i = nrow - 1
    while i >= 0:
        var nz = arow.unsafe_load(i + 1) - adiag.unsafe_load(i) - 1
        var src = adiag.unsafe_load(i) + 1
        var j = 0
        while j < nz:
            bj.unsafe_store(k, acol.unsafe_load(src + j))
            k += 1
            j += 1
        bj.unsafe_store(k, i)
        k += 1
        bdiag.unsafe_store(i, bdiag.unsafe_load(i + 1) + nz + 1)
        i -= 1
    return k


def MatILUFactorSymbolic_SeqAIJ(
    bi: PI, bj: PI, bjlvl: PI, bdiag: PI, arow: PI, acol: PI, r: PI, ic: PI,
    levels: Int, cap: Int, scratch: PI, nrow: Int
) -> Int:
    # `bj` here is the contiguous staging array; MatILUFreeSpaceContiguousLU
    # rearranges it into the final factor layout afterwards.
    """aijfact.c MatILUFactorSymbolic_SeqAIJ, the level-of-fill case.

    Returns the number of factor entries, or its negation when `cap` was too
    small and the caller should retry with that many.
    """
    var nlnk = nrow + 1
    var lnk = scratch
    var lnklvl = scratch.unsafe_offset(nlnk)
    var bt = scratch.unsafe_offset(2 * nlnk)
    lnk.unsafe_store(nrow, nrow)

    bi.unsafe_store(0, 0)
    bdiag.unsafe_store(0, 0)
    var i = 0
    while i < nrow:
        var ri = r.unsafe_load(i)
        var nzi = 0
        var nnz = arow.unsafe_load(ri + 1) - arow.unsafe_load(ri)
        var cols = acol.unsafe_offset(arow.unsafe_load(ri))
        lnk.unsafe_store(i, -1)  # marker: the diagonal has not been seen yet
        nzi += IncompleteLLInit(nnz, cols, nrow, ic, lnk, lnklvl, bt)

        var nzbd = 0
        var prow = lnk.unsafe_load(nrow)
        while prow < i:
            var nb = bdiag.unsafe_load(prow)
            var pcols = bj.unsafe_offset(bi.unsafe_load(prow) + nb + 1)
            var pcolslvl = bjlvl.unsafe_offset(bi.unsafe_load(prow) + nb + 1)
            var lnz = bi.unsafe_load(prow + 1) - bi.unsafe_load(prow) - nb - 1
            # PetscILULLAddSorted offsets by the level of U(prow,prow)
            nzi += IncompleteLLAdd(
                lnz, pcols, Float64(levels), pcolslvl, nrow, lnk, lnklvl, bt,
                lnklvl.unsafe_load(prow), 1
            )
            prow = lnk.unsafe_load(prow)
            nzbd += 1
        bdiag.unsafe_store(i, nzbd)
        if bi.unsafe_load(i) + nzi > cap:
            return -(bi.unsafe_load(i) + nzi)
        bi.unsafe_store(i + 1, bi.unsafe_load(i) + nzi)
        IncompleteLLClean(
            nrow, nrow, nzi, lnk, lnklvl, bj.unsafe_offset(bi.unsafe_load(i)),
            bjlvl.unsafe_offset(bi.unsafe_load(i)), bt
        )
        i += 1
    return bi.unsafe_load(nrow)


def MatICCFactorSymbolic_SeqAIJ(
    ui: PI, uj: PI, ujlvl: PI, udiag: PI, arow: PI, acol: PI, rip: PI, riip: PI,
    levels: Int, cap: Int, scratch: PI, nrow: Int
) -> Int:
    """aijfact.c MatICCFactorSymbolic_SeqAIJ, the level-of-fill case.

    The IC factor keeps only the upper triangle with the diagonal as the last
    entry of each row, so `udiag[i]` points at the diagonal and the strict
    upper triangle runs from `ui[i]` to `udiag[i] - 1`.  Returns the number of
    factor entries, negated when `cap` was too small.
    """
    var nlnk = nrow + 1
    var lnk = scratch
    var lnklvl = scratch.unsafe_offset(nlnk)
    var bt = scratch.unsafe_offset(2 * nlnk)
    var jl = scratch.unsafe_offset(3 * nlnk)
    var il = scratch.unsafe_offset(4 * nlnk)
    var ajtmp = scratch.unsafe_offset(5 * nlnk)
    var i = 0
    while i < nrow:
        jl.unsafe_store(i, nrow)
        il.unsafe_store(i, 0)
        i += 1
    lnk.unsafe_store(nrow, nrow)

    ui.unsafe_store(0, 0)
    var k = 0
    while k < nrow:
        var nzk = 0
        var rk = rip.unsafe_load(k)
        var ncols = arow.unsafe_load(rk + 1) - arow.unsafe_load(rk)
        var ncols_upper = 0
        var j = 0
        while j < ncols:
            var col = acol.unsafe_load(arow.unsafe_load(rk) + j)
            if riip.unsafe_load(col) >= k:  # upper triangular entries only
                ajtmp.unsafe_store(ncols_upper, col)
                ncols_upper += 1
            j += 1
        nzk += IncompleteLLInit(ncols_upper, ajtmp, nrow, riip, lnk, lnklvl, bt)

        var prow = jl.unsafe_load(k)
        while prow < k:
            var nextprow = jl.unsafe_load(prow)
            var jmin = il.unsafe_load(prow) + 1
            var jmax = ui.unsafe_load(prow + 1)
            var ncols2 = jmax - jmin
            if ncols2 > 0:
                var off = jmin - ui.unsafe_load(prow)
                var cols = uj.unsafe_offset(ui.unsafe_load(prow) + off)
                var colslvl = ujlvl.unsafe_offset(ui.unsafe_load(prow) + off)
                # PetscICCLLAddSorted offsets by the level of U(prow,prow)
                nzk += IncompleteLLAdd(
                    ncols2, cols, Float64(levels), colslvl, nrow, lnk, lnklvl, bt,
                    colslvl.unsafe_load(-1), 1
                )
                il.unsafe_store(prow, jmin)
                var cj = cols.unsafe_load(0)
                jl.unsafe_store(prow, jl.unsafe_load(cj))
                jl.unsafe_store(cj, prow)
            prow = nextprow

        if ui.unsafe_load(k) + nzk > cap:
            return -(ui.unsafe_load(k) + nzk)
        IncompleteLLClean(
            nrow, nrow, nzk, lnk, lnklvl, uj.unsafe_offset(ui.unsafe_load(k)),
            ujlvl.unsafe_offset(ui.unsafe_load(k)), bt
        )
        if nzk > 1:
            var cj = uj.unsafe_load(ui.unsafe_load(k) + 1)
            jl.unsafe_store(k, jl.unsafe_load(cj))
            jl.unsafe_store(cj, k)
            il.unsafe_store(k, ui.unsafe_load(k) + 1)
        udiag.unsafe_store(k, ui.unsafe_load(k) + nzk - 1)
        ui.unsafe_store(k + 1, ui.unsafe_load(k) + nzk)
        k += 1
    return ui.unsafe_load(nrow)


# ---------------------------------------------------------------- ordering


def MatGetOrdering_RCM(
    perm: PI, level: PI, visited: PI, queue: PI, arow: PI, acol: PI, nrow: Int
):
    """Reverse Cuthill-McKee, the ordering PETSc uses by default for ILU.

    A breadth-first sweep from each unvisited node, visiting neighbours in
    increasing index order, with each component reversed.  `level`, `visited`
    and `queue` are `nrow` scratch entries; disconnected graphs give one
    reversed component per connected piece, in the order they are found.

    Divergence: PETSc's Cuthill-McKee visits each node's neighbours in
    increasing *degree* order, which is the part that decides the fill.  This
    port visits them in increasing index, so the ordering it produces is a
    valid permutation but not always the same one PETSc picks.
    """
    var i = 0
    while i < nrow:
        level.unsafe_store(i, arow.unsafe_load(i + 1) - arow.unsafe_load(i))
        visited.unsafe_store(i, 0)
        i += 1

    var filled = 0
    var start = 0
    while start < nrow:
        if visited.unsafe_load(start) == 0:
            visited.unsafe_store(start, 1)
            queue.unsafe_store(0, start)
            var head = 0
            var tail = 1
            while head < tail:
                var node = queue.unsafe_load(head)
                head += 1
                var j = arow.unsafe_load(node)
                var jend = arow.unsafe_load(node + 1)
                while j < jend:
                    var c = acol.unsafe_load(j)
                    if visited.unsafe_load(c) == 0:
                        visited.unsafe_store(c, 1)
                        queue.unsafe_store(tail, c)
                        tail += 1
                    j += 1
            # reverse this component, then append it to the result
            i = 0
            while i < tail // 2:
                var t = queue.unsafe_load(i)
                queue.unsafe_store(i, queue.unsafe_load(tail - 1 - i))
                queue.unsafe_store(tail - 1 - i, t)
                i += 1
            i = 0
            while i < tail:
                perm.unsafe_store(filled + i, queue.unsafe_load(i))
                i += 1
            filled += tail
        start += 1


def MatILUFreeSpaceContiguousLU(
    bjsrc: PI, bi: PI, bdiag: PI, bj: PI, n: Int
):
    """freespace.c PetscFreeSpaceContiguous_LU.

    The symbolic phase lays row i out contiguously (its L entries, the
    diagonal, then its U entries) and records the L count in `bdiag[i]`.  The
    factor has to be stored the other way round — all L rows in increasing
    order, then all U rows in decreasing order, each U row ending with its
    diagonal — and this is the pass that rearranges one into the other.

    `bjsrc` and `bj` must be distinct: the copy is not safe in place.
    """
    var bi_temp = bi.unsafe_load(n)
    var row = 0
    var src = 0
    var nnzL = bdiag.unsafe_load(0)
    while row < n:
        var nnz = bi.unsafe_load(row + 1) - bi.unsafe_load(row)
        if row == 0:
            bi.unsafe_store(row, 0)
        else:
            bi.unsafe_store(row, bi.unsafe_load(row - 1) + nnzL)
        nnzL = bdiag.unsafe_load(row)
        # L part
        var k = 0
        var dst = bi.unsafe_load(row)
        while k < nnzL:
            bj.unsafe_store(dst + k, bjsrc.unsafe_load(src + k))
            k += 1
        # diagonal entry
        bdiag.unsafe_store(row, bi_temp - 1)
        bj.unsafe_store(bi_temp - 1, row)
        # U part, excluding the diagonal
        var nnzU = nnz - nnzL
        bi_temp -= nnzU
        nnzU -= 1
        k = 0
        dst = bi_temp
        while k < nnzU:
            bj.unsafe_store(dst + k, bjsrc.unsafe_load(src + nnzL + 1 + k))
            k += 1
        src += nnz
        row += 1
    if n:
        bi.unsafe_store(n, bi.unsafe_load(n - 1) + nnzL)
        bdiag.unsafe_store(n, bdiag.unsafe_load(n - 1) - 1)


def MatICCFreeSpaceContiguousCholesky(ui: PI, udiag: PI, uj: PI, n: Int):
    """freespace.c PetscFreeSpaceContiguous_Cholesky.

    The linked list holds the columns of U(k,:) in ascending order, and the
    diagonal k is the smallest of them, so the staging array has the diagonal
    first.  The factor wants it last, with the strict upper triangle before
    it.  The shift is toward lower addresses, so it is safe in place.
    """
    var row = 0
    while row < n:
        var base = ui.unsafe_load(row)
        var nnz = ui.unsafe_load(row + 1) - base - 1
        udiag.unsafe_store(row, ui.unsafe_load(row + 1) - 1)
        var diag = uj.unsafe_load(base)
        var j = 0
        while j < nnz:
            uj.unsafe_store(base + j, uj.unsafe_load(base + j + 1))
            j += 1
        uj.unsafe_store(base + nnz, diag)
        row += 1

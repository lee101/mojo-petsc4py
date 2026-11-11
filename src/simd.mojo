"""SIMD width for the elementwise kernels in `simdvec.mojo`.

Every kernel in that module is a contiguous pass over a flat `Float64` array,
so all of them have the same loop shape: a vector body handling `SW` elements
per iteration and a scalar tail.  `SW` is the SIMD width of `Float64` on the
build target, read once at compile time.

No manual alignment peel is needed.  Buffers arrive from Python as `numpy`
arrays, whose address is 16-byte aligned rather than `SW` aligned, but the
`unsafe_load[width=SW]` form lowers to an unaligned vector load (`vmovupd`) and
the store likewise; on aligned data those cost the same as the aligned forms,
and on a misaligned buffer they cost what hardware unaligned access costs
rather than what a runtime peel loop would cost on every call.
"""

from std.sys import simd_width_of

comptime SW = simd_width_of[DType.float64]()

# A reduction accumulates a block of `BLK` elements into four independent SIMD
# accumulators.  One accumulator sustains one load per three cycles (one FMA
# latency), so on a 2.3 GHz Haswell core it streams 32 bytes every 3 cycles,
# about 25 GB/s; an out-of-cache vector needs four loads in flight to reach
# memory bandwidth.
#
# `BLK` is the width that is measured rather than inferred: one `BLK`-element
# block per iteration is the form whose four accumulators appear as four
# registers in the disassembly and whose results were checked against numpy for
# lengths on both sides of every boundary.  A wider block buys nothing here --
# 16 loads in flight on this core streams the same 33 GB/s as four once the
# array is out of cache -- so the narrower form is kept because it is simpler.
comptime PV = 4
comptime BLK = PV * SW

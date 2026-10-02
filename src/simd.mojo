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
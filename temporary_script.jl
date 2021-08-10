# this script should be removed when the alternative MO kernelmatrix is accepted or rejected. 

using KernelFunctions, BenchmarkTools
using LinearAlgebra

mrank = 1
dims = (in=5, out=3)
x = [rand(dims.in) for _ in 1:20]

xMOF = KernelFunctions.MOInputIsotopicByFeatures(x, dims.out)
xMOO = KernelFunctions.MOInputIsotopicByOutputs(x, dims.out)

indk = IndependentMOKernel(GaussianKernel())

Kind1 = kernelmatrix(indk, xMOF, xMOF)
Kind2 = kernelmatrix2(indk, xMOF, xMOF)

Kind1 ≈ Kind2
# true

@benchmark kernelmatrix($indk, $xMOF, $xMOF)
# BenchmarkTools.Trial: 756 samples with 1 evaluation.
#  Range (min … max):  6.186 ms …  11.470 ms  ┊ GC (min … max): 0.00% … 35.30%
#  Time  (median):     6.423 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   6.614 ms ± 806.792 μs  ┊ GC (mean ± σ):  2.76% ±  7.77%

#    ▅▇█▇▃                                                       
#   ▆██████▅▆▄▅▄▁▄▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁▆██▇▇ ▇
#   6.19 ms      Histogram: log(frequency) by time        10 ms <

#  Memory estimate: 2.34 MiB, allocs estimate: 60060.

@benchmark kernelmatrix2($indk, $xMOF, $xMOF)
# BenchmarkTools.Trial: 10000 samples with 5 evaluations.
#  Range (min … max):  6.162 μs … 341.226 μs  ┊ GC (min … max):  0.00% … 96.29%
#  Time  (median):     6.947 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   8.235 μs ±  16.802 μs  ┊ GC (mean ± σ):  12.37% ±  5.92%

#    ▁▄▆▇██▇▆▅▄▃▂▁▁▁                                            ▂
#   ▅██████████████████▇▇▆▇▅▆▆▆▅▅▆▅▅▃▄▅▅▅▁▁▃▁▃▅▅▆▆▆▇▇▇▇▇▇▇██▇▇▇ █
#   6.16 μs      Histogram: log(frequency) by time      13.7 μs <

#  Memory estimate: 34.89 KiB, allocs estimate: 7.

A = randn(dims.out, mrank)
B = A * transpose(A) + Diagonal(rand(dims.out))

ickernel = IntrinsicCoregionMOKernel(GaussianKernel(), B)

Kic1 = kernelmatrix(ickernel, xMOF, xMOF)
Kic2 = kernelmatrix2(ickernel, xMOF, xMOF)

Kic1 ≈ Kic2
#true

@benchmark kernelmatrix($ickernel, $xMOF, $xMOF)
# BenchmarkTools.Trial: 1874 samples with 1 evaluation.
#  Range (min … max):  2.522 ms …   5.424 ms  ┊ GC (min … max): 0.00% … 51.13%
#  Time  (median):     2.601 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   2.666 ms ± 369.030 μs  ┊ GC (mean ± σ):  2.01% ±  7.04%

#   ▂█▆▁  ▁                                                      
#   ███████▅▄▅▅▃▁▁▃▅▁▁▁▃▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▇ █
#   2.52 ms      Histogram: log(frequency) by time       570 ms <

#  Memory estimate: 985.47 KiB, allocs estimate: 39639.

@benchmark kernelmatrix2($ickernel, $xMOF, $xMOF)
# BenchmarkTools.Trial: 10000 samples with 5 evaluations.
#  Range (min … max):  6.152 μs … 322.002 μs  ┊ GC (min … max):  0.00% … 96.14%
#  Time  (median):     6.676 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   8.100 μs ±  16.959 μs  ┊ GC (mean ± σ):  12.53% ±  5.85%

#    ▄▇██▇▅▄▃▂▁▁                                        ▁▁▁▁    ▂
#   ▆██████████████▇▇▇▆▆▇▇▇▇▆▇▆▄▅▄▃▆▁▅▄▅▆▆▅▅▄▅▄▆▇▇█▇▇▇▇██████▇▇ █
#   6.15 μs      Histogram: log(frequency) by time      13.5 μs <

#  Memory estimate: 34.77 KiB, allocs estimate: 6.

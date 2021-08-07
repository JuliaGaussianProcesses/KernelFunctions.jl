# this script should be removed when the alternative MO kernelmatrix is accepted or rejected. 

using KernelFunctions, BenchmarkTools

rank = 1
dims = (in=5, out=3)
x = [rand(dims.in) for _ in 1:20]

xMOF = KernelFunctions.MOInputIsotopicByFeatures(x, dims.out)
xMOO  = KernelFunctions.MOInputIsotopicByOutputs(x, dims.out)

indk = IndependentMOKernel(GaussianKernel())

Kind1 = kernelmatrix(indk, xMOF, xMOF)
Kind2 = kernelmatrix2(indk, xMOF, xMOF)

Kind1 ≈ Kind2

@benchmark kernelmatrix($indk, $xMOF, $xMOF)
@benchmark kernelmatrix2($indk, $xMOF, $xMOF)

A = randn(dims.out, rank)
B = A * transpose(A) + Diagonal(rand(dims.out))

ickernel = IntrinsicCoregionMOKernel(GaussianKernel(), B)

Kic1 = kernelmatrix(ickernel, xMOF, xMOF)
Kic2 = kernelmatrix2(ickernel, xMOF, xMOF)

Kic1 ≈ Kic2

@benchmark kernelmatrix($ickernel, $xMOF, $xMOF)
@benchmark kernelmatrix2($ickernel, $xMOF, $xMOF)
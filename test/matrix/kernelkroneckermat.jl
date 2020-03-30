using KernelFunctions
using Kronecker

using Random
using Test

rng = MersenneTwister(123456)
k = SqExponentialKernel()
x = range(0, 1; length = 10)
X = vcat(collect.(Iterators.product(x, x))'...)

@test all(collect(kernelkronmat(k, collect(x), 2)) .≈ kernelmatrix(k, X, obsdim = 1))
@test all(collect(kernelkronmat(k, [x,x])) .≈ kernelmatrix(k, X, obsdim = 1))
@test_throws AssertionError kernelkronmat(LinearKernel(), collect(x), 2)

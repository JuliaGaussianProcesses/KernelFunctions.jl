using KernelFunctions

using Test

k = SqExponentialKernel()
@test length(k) == 1
@test iterate(k) == (k, nothing)
@test iterate(k, 1) == nothing

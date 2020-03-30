using KernelFunctions
using KernelFunctions: metric

using LinearAlgebra
using Random
using Test

rng = MersenneTwister(123456)
x = rand(rng) * 2
v1 = rand(rng, 3)
v2 = rand(rng, 3)

k = ExponentiatedKernel()
@test kappa(k, x) ≈ exp(x)
@test kappa(k, -x) ≈ exp(-x)
@test k(v1, v2) ≈ exp(dot(v1, v2))
@test metric(ExponentiatedKernel()) == KernelFunctions.DotProduct()

using KernelFunctions

using Random
using Test

rng = MersenneTwister(123456)
x = rand(rng) * 2
v1 = rand(rng, 3)
v2 = rand(rng, 3)

k1 = LinearKernel()
k2 = SqExponentialKernel()
k3 = RationalQuadraticKernel()
X = rand(rng, 2, 2)

w = [2.0,0.5]
k = KernelSum([k1,k2], w)
ks1 = 2.0 * k1
ks2 = 0.5 * k2
@test length(k) == 2
@test kappa(k, v1, v2) == kappa(2.0 * k1 + 0.5 * k2, v1, v2)
@test kappa(k + k3, v1, v2) ≈ kappa(k3 + k, v1, v2)
@test kappa(k1 + k2, v1, v2) == kappa(KernelSum([k1,k2]), v1, v2)
@test kappa(k + ks1, v1, v2) ≈ kappa(ks1 + k, v1, v2)
@test kappa(k + k, v1, v2) == kappa(KernelSum([k1,k2,k1,k2], vcat(w, w)), v1, v2)


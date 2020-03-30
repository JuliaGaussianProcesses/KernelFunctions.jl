using KernelFunctions
using Distances

using Test

P = rand(3, 3)
k = MahalanobisKernel(P)

x = 2 * rand()
@test kappa(k, x) == exp(-x)
@test kappa(ExponentialKernel(), x) == kappa(k, x)

v1 = rand(3)
v2 = rand(3)
@test k(v1, v2) ≈ exp(-sqmahalanobis(v1, v2, k.P))

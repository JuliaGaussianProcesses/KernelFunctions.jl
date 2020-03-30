using KernelFunctions
using KernelFunctions: metric
using Distances
using SpecialFunctions: besselk, gamma

using LinearAlgebra
using Random
using Test

rng = MersenneTwister(123456)
x = rand(rng) * 2
v1 = rand(rng, 3)
v2 = rand(rng, 3)
@testset "MaternKernel" begin
    ν = 2.0
    k = MaternKernel(ν = ν)
    matern(x, ν) = 2^(1 - ν) / gamma(ν) * (sqrt(2ν) * x)^ν * besselk(ν, sqrt(2ν) * x)
    @test MaternKernel(nu = ν).ν == [ν]
    @test kappa(k, x) ≈ matern(x, ν)
    @test kappa(k, 0.0) == 1.0
    @test kappa(MaternKernel(ν = ν), x) == kappa(k, x)
    @test metric(MaternKernel()) == Euclidean()
    @test metric(MaternKernel(ν = 2.0)) == Euclidean()
end
@testset "Matern32Kernel" begin
    k = Matern32Kernel()
    @test kappa(k, x) ≈ (1 + sqrt(3) * x)exp(-sqrt(3) * x)
    @test k(v1, v2) ≈ (1 + sqrt(3) * norm(v1 - v2))exp(-sqrt(3) * norm(v1 - v2))
    @test kappa(Matern32Kernel(), x) == kappa(k, x)
    @test metric(Matern32Kernel()) == Euclidean()
end
@testset "Matern52Kernel" begin
    k = Matern52Kernel()
    @test kappa(k, x) ≈ (1 + sqrt(5) * x + 5 / 3 * x^2)exp(-sqrt(5) * x)
    @test k(v1, v2) ≈ (1 + sqrt(5) * norm(v1 - v2) + 5 / 3 * norm(v1 - v2)^2)exp(-sqrt(5) * norm(v1 - v2))
    @test kappa(Matern52Kernel(), x) == kappa(k, x)
    @test metric(Matern52Kernel()) == Euclidean()
end
@testset "Coherence Materns" begin
    @test kappa(MaternKernel(ν = 0.5), x) ≈ kappa(ExponentialKernel(), x)
    @test kappa(MaternKernel(ν = 1.5), x) ≈ kappa(Matern32Kernel(), x)
    @test kappa(MaternKernel(ν = 2.5), x) ≈ kappa(Matern52Kernel(), x)
end


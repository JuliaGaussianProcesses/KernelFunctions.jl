using KernelFunctions
using KernelFunctions: kappa
using Flux: params
import Zygote, ForwardDiff, ReverseDiff
using Test, LinearAlgebra, Random
using FiniteDifferences

include("utils_AD.jl")

dims = [3, 3]
ν = 3.0

rng = MersenneTwister(42)

A = rand(rng, dims...)
B = rand(rng, dims...)
K = [zeros(dims[1], dims[1]), zeros(dims[2], dims[2])]

x = rand(rng, dims[1])
y = rand(rng, dims[1])

l = rand(rng)
vl = l * ones(dims[1])

kernels = [
    SqExponentialKernel(),
    ExponentialKernel(),
    MaternKernel(ν = ν),
    # transform(SqExponentialKernel(), l),
    # transform(SqExponentialKernel(), vl),
    # ExponentiatedKernel() + LinearKernel(),
    # 2.0 * PolynomialKernel() * Matern32Kernel(),
]

ds = log.([eps(), rand(rng)])

testfunction(k, A, B, dim) = det(kernelmatrix(k, A, B, obsdim = dim))
testfunction(k, A, dim) = det(kernelmatrix(k, A, obsdim = dim))
ADs = [:Zygote, :ForwardDiff, :ReverseDiff]


## Test kappa functions

@testset "Kappa functions" begin
    for k in kernels[isa.(kernels, KernelFunctions.SimpleKernel)]
        @testset "$k" begin
            @test_nowarn gradient(Val(:FiniteDiff), x -> kappa(k, exp(x[1])), ds[1]) # Check FiniteDiff does the right thing
            for AD in ADs
                @testset "$AD" begin
                    for d in ds
                        @test_nowarn gradient(Val(AD), x -> kappa(k, exp(x[1])), [d])
                        @test gradient(Val(AD), x -> kappa(k, exp(x[1])), [d]) ≈ gradient(Val(:FiniteDiff), x -> kappa(k, exp(x[1])), [d]) atol=1e-8
                    end
                end
            end
        end
    end
end

@testset "Kernel evaluations" begin
    for k in kernels
        @testset "$k" begin
            for AD in ADs
                @test_nowarn gradient(Val(:FiniteDiff), x -> k(x, y), x)
                @testset "$AD" begin
                    for d in ds
                        @test_nowarn gradient(Val(AD), x -> k(x, y), x)
                        @test gradient(Val(AD), x -> k(x, y), x) ≈ gradient(Val(:FiniteDiff), x -> k(x, y), x) atol=1e-8
                    end
                end
            end
        end
    end
end

@testset "Kernel Matrices" begin
    for k in kernels
        @testset "$k" begin
            for AD in ADs
                # @test_nowarn gradient(Val(:FiniteDiff), x -> k(x, y), )
                @testset "$AD" begin
                    for dim in [1,2]
                        @test_nowarn gradient(Val(AD), x -> testfunction(k, x, dim), A)
                        @test_nowarn gradient(Val(AD), x -> testfunction(k, x, B, dim), A)
                        @test gradient(Val(AD), x -> testfunction(k, x, B, dim), A) ≈ gradient(Val(:FiniteDiff), x -> testfunction(k, x, B, dim), A) atol=1e-8
                        @test gradient(Val(AD), x -> testfunction(k, x, dim), A) ≈ gradient(Val(:FiniteDiff), x -> testfunction(k, x, dim), A) atol=1e-8
                    end
                end
            end
        end
    end
end

@testset "Params differentiation" begin
    for k in kernels
        @testset "$k" begin
            ps = params(k)
            @test_nowarn gradient(Val(:Zygote), () -> k(x, y), ps)
        end
    end
end

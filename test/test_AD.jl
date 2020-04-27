using KernelFunctions
using Zygote, ForwardDiff
using Test, LinearAlgebra
using FiniteDifferences

dims = [10,5]

A = rand(dims...)
B = rand(dims...)
K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
kernels_noparams = [:SqExponentialKernel,:ExponentialKernel,:GammaExponentialKernel,
 :MaternKernel,:Matern32Kernel,:Matern52Kernel,
 :LinearKernel,:PolynomialKernel,
 :RationalQuadraticKernel,:GammaRationalQuadraticKernel,
 :ExponentiatedKernel]
l = 2.0
ds = [0.0,3.0]
vl = l*ones(dims[1])
testfunction(k,A,B) = det(kernelmatrix(k,A,B))
testfunction(k,A) = det(kernelmatrix(k,A))
ADs = [:Zygote,:ForwardDiff]

## Test kappa functions
@testset "Kappa functions" begin
    for AD in ADs
        @testset "$AD" begin
            for k in kernels_noparams
                for d in ds
                    @eval begin @test kappa_AD(Val(Symbol($AD)),$k(),$d) ≈ kappa_fdm($k(),$d) atol=1e-8 end
                end
            end
            # Linear -> C
            # Polynomial -> C,D
            # Gamma (etc) -> gamma
            #
        end
    end
end

@testset "Transform Operations" begin
    for AD in ADs
        @testset "$AD" begin
            @eval begin
            # Scale Transform
            transform_AD(Val(Symbol($AD)),ScaleTransform(l),A)
            # ARD Transform
            transform_AD(Val(Symbol($AD)),ARDTransform(vl),A)
            # Linear transform
            transform_AD(Val(Symbol($AD)), LinearTransform(rand(2,10)),A)
            # Chain Transform
            # transform_AD(Val(Symbol($AD)), LinearTransform, A)
            end
        end
    end
end

##TODO Eventually store real results in file
@testset "Zygote Automatic Differentiation test" begin
    @testset "ARD" begin
        for k in kernels
            @testset "$k" begin
                @test all(isapprox.(Zygote.gradient(x->testfunction(k(x),A,B),vl)[1], ForwardDiff.gradient(x->testfunction(k(x),A,B),vl)))
                @test  all(isapprox.(Zygote.gradient(x->testfunction(k(vl),x,B),A)[1],ForwardDiff.gradient(x->testfunction(k(vl),x,B),A)))
                @test all(isapprox.(Zygote.gradient(x->testfunction(k(x),A),vl)[1],ForwardDiff.gradient(x->testfunction(k(x),A),vl)))
                @test all(isapprox.(Zygote.gradient(x->testfunction(k(vl),x),A)[1],ForwardDiff.gradient(x->testfunction(k(vl),x),A)))
            end
        end
    end
    @testset "ISO" begin
        for k in kernels
            @testset "$k" begin
                @test all(isapprox.(Zygote.gradient(x->testfunction(k(x),A,B),l)[1],ForwardDiff.gradient(x->testfunction(k(x[1]),A,B),[l])[1]))
                @test all(isapprox.(Zygote.gradient(x->testfunction(k(l),x,B),A)[1],ForwardDiff.gradient(x->testfunction(k(l),x,B),A)))
                @test all(isapprox.(Zygote.gradient(x->testfunction(k(x),A),l)[1],ForwardDiff.gradient(x->testfunction(k(x[1]),A),[l])))
                @test all(isapprox.(Zygote.gradient(x->testfunction(k(l),x),A)[1],ForwardDiff.gradient(x->testfunction(k(l[1]),x),A)))
            end
        end
    end
end

@testset "ForwardDiff AutomaticDifferentation test" begin
    @testset "ARD" begin
        for k in kernels
            @test_nowarn ForwardDiff.gradient(x->testfunction(k(x),A,B),vl)
            @test_nowarn ForwardDiff.gradient(x->testfunction(k(vl),x,B),A)
            @test_nowarn ForwardDiff.gradient(x->testfunction(k(x),A),vl)
            @test_nowarn ForwardDiff.gradient(x->testfunction(k(vl),x),A)
        end
    end
    @testset "ISO" begin
        for k in kernels
            @test_nowarn ForwardDiff.gradient(x->testfunction(k(x[1]),A,B),[l])
            @test_nowarn ForwardDiff.gradient(x->testfunction(k(l),x,B),A)
            @test_nowarn ForwardDiff.gradient(x->testfunction(k(x[1]),A),[l])
            @test_nowarn ForwardDiff.gradient(x->testfunction(k(l[1]),x),A)
        end
    end
end


@testset "Tracker AutomaticDifferentation test" begin
    @testset "ARD" begin
        for k in kernels
            @test_broken all(Tracker.gradient(x->testfunction(k(x),A,B),vl)[1] .≈ ForwardDiff.gradient(x->testfunction(k(x),A,B),vl))
            @test_broken all(Tracker.gradient(x->testfunction(k(vl),x,B),A)[1] .≈ ForwardDiff.gradient(x->testfunction(k(vl),x,B),A))
            @test_broken all(Tracker.gradient(x->testfunction(k(x),A),vl)[1] .≈  ForwardDiff.gradient(x->testfunction(k(x),A),vl))
            @test_broken all.(Tracker.gradient(x->testfunction(k(vl),x),A) .≈ ForwardDiff.gradient(x->testfunction(k(vl),x),A))
        end
    end
    @testset "ISO" begin
        for k in kernels
            @test_broken Tracker.gradient(x->testfunction(k(x[1]),A,B),[l])
            @test_broken Tracker.gradient(x->testfunction(k(l),x,B),A)
            @test_broken Tracker.gradient(x->testfunction(k(x[1]),A),[l])
            @test_broken Tracker.gradient(x->testfunction(k(l),x),A)

        end
    end
end

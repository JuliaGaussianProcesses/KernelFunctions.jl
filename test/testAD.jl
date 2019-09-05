using KernelFunctions
using Zygote, ForwardDiff, Tracker
using Test

dims = [10,5]

A = rand(dims...)
B = rand(dims...)
K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
kernels = [SquaredExponentialKernel]
l = 2.0
vl = l*ones(dims[1])
testfunction(k,A,B) = sum(kernelmatrix(k,A,B))
testfunction(k,A) = sum(kernelmatrix(k,A))

testfunction(SquaredExponentialKernel(vl),A)
##Eventually store real results in file
@testset "Zygote Automatic Differentiation test" begin
    @testset "ARD" begin
        for k in kernels
            @test all(isapprox.(Zygote.gradient(x->testfunction(k(x),A,B),vl)[1], ForwardDiff.gradient(x->testfunction(k(x),A,B),vl)))
            @test  all(isapprox.(Zygote.gradient(x->testfunction(k(vl),x,B),A)[1],ForwardDiff.gradient(x->testfunction(k(vl),x,B),A)))
            @test all(isapprox.(Zygote.gradient(x->testfunction(k(x),A),vl)[1],ForwardDiff.gradient(x->testfunction(k(x),A),vl)))
            @test all(isapprox.(Zygote.gradient(x->testfunction(k(vl),x),A)[1],ForwardDiff.gradient(x->testfunction(k(vl),x),A)))
        end
    end
    @testset "ISO" begin
        for k in kernels
            @test all(isapprox.(Zygote.gradient(x->testfunction(k(x),A,B),l)[1],ForwardDiff.gradient(x->testfunction(k(x[1]),A,B),[l])[1]))
            @test all(isapprox.(Zygote.gradient(x->testfunction(k(l),x,B),A)[1],ForwardDiff.gradient(x->testfunction(k(l),x,B),A)))
            @test all(isapprox.(Zygote.gradient(x->testfunction(k(x),A),l)[1],ForwardDiff.gradient(x->testfunction(k(x[1]),A),[l])))
            @test all(isapprox.(Zygote.gradient(x->testfunction(k(l),x),A)[1],ForwardDiff.gradient(x->testfunction(k(l[1]),x),A)))
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

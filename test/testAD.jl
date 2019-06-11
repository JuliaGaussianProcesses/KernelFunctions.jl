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


#For debugging

## Zygote
# Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),vl)
# Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A),vl)
# Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),l)
# Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A),l)

## Tracker
Tracker.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),vl)
Tracker.gradient(x->testfunction(SquaredExponentialKernel(x),A),vl)
Tracker.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),l)
Tracker.gradient(x->testfunction(SquaredExponentialKernel(x),A),l)


## ForwardDiff
ForwardDiff.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),vl) #✓
ForwardDiff.gradient(x->testfunction(SquaredExponentialKernel(x),A),vl) #✓
ForwardDiff.gradient(x->testfunction(SquaredExponentialKernel(x[1]),A,B),[l])
ForwardDiff.gradient(x->testfunction(SquaredExponentialKernel(x[1]),A),[l])
##Eventually store real results in file

@testset "Zygote Automatic Differentiation test" begin
    @testset "ARD" begin
        for k in kernels
            @test_broken Zygote.gradient(x->testfunction(k(x),A,B),vl)
            @test_broken Zygote.gradient(x->testfunction(k(vl),x,B),A)
            @test_broken Zygote.gradient(x->testfunction(k(x),A),vl)
            @test_broken Zygote.gradient(x->testfunction(k(vl),x),A)
        end
    end
    @testset "ISO" begin
        for k in kernels
            @test_broken Zygote.gradient(x->testfunction(k(x),A,B),l)
            @test_broken Zygote.gradient(x->testfunction(k(l),x,B),A)
            @test_broken Zygote.gradient(x->testfunction(k(x),A),l)
            @test_broken Zygote.gradient(x->testfunction(k(l),x),A)

        end
    end
end

@testset "Tracker AutomaticDifferentation test" begin
    @testset "ARD" begin
        for k in kernels
            @test_nowarn Tracker.gradient(x->testfunction(k(x),A,B),vl)
            @test_broken Tracker.gradient(x->testfunction(k(vl),x,B),A)
            @test_nowarn Tracker.gradient(x->testfunction(k(x),A),vl)
            @test_broken Tracker.gradient(x->testfunction(k(vl),x),A)
        end
    end
    @testset "ISO" begin
        for k in kernels
            @test_nowarn Tracker.gradient(x->testfunction(k(x[1]),A,B),[l])
            @test_broken Tracker.gradient(x->testfunction(k(l),x,B),A)
            @test_nowarn Tracker.gradient(x->testfunction(k(x[1]),A),[l])
            @test_broken Tracker.gradient(x->testfunction(k(l),x),A)

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

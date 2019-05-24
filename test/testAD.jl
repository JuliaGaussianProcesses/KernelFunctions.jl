using Zygote, ForwardDiff, Tracker

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
Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),vl)
Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A),vl)
Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A,B),l)
Zygote.gradient(x->testfunction(SquaredExponentialKernel(x),A),l)

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
            @test Zygote.gradient(x->testfunction(k(x),A,B),vl)
            @test Zygote.gradient(x->testfunction(k(vl),x,B),A)
            @test Zygote.gradient(x->testfunction(k(x),A),vl)
            @test Zygote.gradient(x->testfunction(k(vl),x),A)
        end
    end
    @testset "ISO" begin
        for k in kernels
            @test Zygote.gradient(x->testfunction(k(x),A,B),l)
            @test Zygote.gradient(x->testfunction(k(l),x,B),A)
            @test Zygote.gradient(x->testfunction(k(x),A),l)
            @test Zygote.gradient(x->testfunction(k(l),x),A)

        end
    end
end

@testset "Tracker AutomaticDifferentation test" begin
    @testset "ARD" begin
        for k in kernels
            @test Tracker.gradient(x->testfunction(k(x),A,B),vl)
            @test Tracker.gradient(x->testfunction(k(vl),x,B),A)
            @test Tracker.gradient(x->testfunction(k(x),A),vl)
            @test Tracker.gradient(x->testfunction(k(vl),x),A)
        end
    end
    @testset "ISO" begin
        for k in kernels
            @test Tracker.gradient(x->testfunction(k(x[1]),A,B),[l])
            @test Tracker.gradient(x->testfunction(k(l),x,B),A)
            @test Tracker.gradient(x->testfunction(k(x),A),[l])
            @test Tracker.gradient(x->testfunction(k(l[1]),x),A)

        end
    end
end


@testset "ForwardDiff AutomaticDifferentation test" begin
    @testset "ARD" begin
        for k in kernels
            @test ForwardDiff.gradient(x->testfunction(k(x),A,B),vl)
            @test ForwardDiff.gradient(x->testfunction(k(vl),x,B),A)
            @test ForwardDiff.gradient(x->testfunction(k(x),A),vl)
            @test ForwardDiff.gradient(x->testfunction(k(vl),x),A)
        end
    end
    @testset "ISO" begin
        for k in kernels
            @test ForwardDiff.gradient(x->testfunction(k(x[1]),A,B),[l])
            @test ForwardDiff.gradient(x->testfunction(k(l),x,B),A)
            @test ForwardDiff.gradient(x->testfunction(k(x),A),[l])
            @test ForwardDiff.gradient(x->testfunction(k(l[1]),x),A)

        end
    end
end

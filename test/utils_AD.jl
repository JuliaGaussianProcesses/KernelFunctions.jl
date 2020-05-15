
FDM = FiniteDifferences.central_fdm(5, 1)

function gradient(::Val{:Zygote}, f::Function, args)
    g = first(Zygote.gradient(f, args))
    if isnothing(g)
        return zeros(size(args)) # To respect the same output as other ADs
    else
        return g
    end
end

function gradient(::Val{:ForwardDiff}, f::Function, args)
    ForwardDiff.gradient(f, args)
end

function gradient(::Val{:ReverseDiff}, f::Function, args)
    ReverseDiff.gradient(f, args)
end

function gradient(::Val{:FiniteDiff}, f::Function, args)
    first(FiniteDifferences.grad(FDM, f, args))
end


testfunction(k, A, B, dim) = sum(kernelmatrix(k, A, B, obsdim = dim))
testfunction(k, A, dim) = sum(kernelmatrix(k, A, obsdim = dim))

function test_AD(kernelname::String, kernelfunction, args = nothing; ADs = [:Zygote, :ForwardDiff, :ReverseDiff], dims = [3, 3])
    test_fd = test_FiniteDiff(kernelname, kernelfunction, args, dims)
    if !test_fd.anynonpass
        for AD in ADs
            test_AD(AD, kernelname, kernelfunction, args, dims)
        end
    end
end

function test_FiniteDiff(kernelname, kernelfunction, args = nothing, dims = [3, 3])
    # Init arguments :
    k = if args === nothing
        kernelfunction()
    else
        kernelfunction(args)
    end
    rng = MersenneTwister(42)
    @testset "FiniteDifferences with $(kernelname)" begin
        if k isa SimpleKernel
            for d in log.([eps(), rand(rng)])
                @test_nowarn gradient(Val(:FiniteDiff), x -> kappa(k, exp(first(x))), [d])
            end
        end
        ## Testing Kernel Functions
        x = rand(rng, dims[1])
        y = rand(rng, dims[1])
        @test_nowarn gradient(Val(:FiniteDiff), x -> k(x, y), x)
        if !(args === nothing)
            @test_nowarn gradient(Val(:FiniteDiff), p -> kernelfunction(p)(x, y), args)
        end
        ## Testing Kernel Matrices
        A = rand(rng, dims...)
        B = rand(rng, dims...)
        for dim in 1:2
            @test_nowarn gradient(Val(:FiniteDiff), a -> testfunction(k, a, dim), A)
            @test_nowarn gradient(Val(:FiniteDiff), a -> testfunction(k, a, B, dim), A)
            @test_nowarn gradient(Val(:FiniteDiff), b -> testfunction(k, A, b, dim), B)
            if !(args === nothing)
                @test_nowarn gradient(Val(:FiniteDiff), p -> testfunction(kernelfunction(p), A, B, dim), args)
            end
        end
    end
end

function test_AD(AD, kernelname, kernelfunction, args = nothing, dims = [3, 3])
    @testset "Testing $(kernelname) with AD : $(AD)" begin
        # Test kappa function
        k = if args === nothing
            kernelfunction()
        else
            kernelfunction(args)
        end
        rng = MersenneTwister(42)
        if k isa SimpleKernel
            for d in log.([eps(), rand(rng)])
                @test gradient(Val(AD), x -> kappa(k, exp(x[1])), [d]) ≈ gradient(Val(:FiniteDiff), x -> kappa(k, exp(x[1])), [d]) atol=1e-8
            end
        end
        # Testing kernel evaluations
        x = rand(rng, dims[1])
        y = rand(rng, dims[1])
        @test gradient(Val(AD), x -> k(x, y), x) ≈ gradient(Val(:FiniteDiff), x -> k(x, y), x) atol=1e-8
        @test gradient(Val(AD), y -> k(x, y), y) ≈ gradient(Val(:FiniteDiff), y -> k(x, y), y) atol=1e-8
        if !(args === nothing)
            @test gradient(Val(AD), p -> kernelfunction(p)(x,y), args) ≈ gradient(Val(:FiniteDiff), p -> kernelfunction(p)(x, y), args) atol=1e-8
        end
        # Testing kernel matrices
        A = rand(rng, dims...)
        B = rand(rng, dims...)
        for dim in 1:2
            @test gradient(Val(AD), x -> testfunction(k, x, dim), A) ≈ gradient(Val(:FiniteDiff), x -> testfunction(k, x, dim), A) atol=1e-8
            @test gradient(Val(AD), a -> testfunction(k, a, B, dim), A) ≈ gradient(Val(:FiniteDiff), a -> testfunction(k, a, B, dim), A) atol=1e-8
            @test gradient(Val(AD), b -> testfunction(k, A, b, dim), B) ≈ gradient(Val(:FiniteDiff), b -> testfunction(k, A, b, dim), B) atol=1e-8
            if !(args === nothing)
                @test gradient(Val(AD), p -> testfunction(kernelfunction(p), A, dim), args) ≈ gradient(Val(:FiniteDiff), p -> testfunction(kernelfunction(p), A, dim), args) atol=1e-8
            end
        end
    end
end

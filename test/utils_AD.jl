
const FDM = FiniteDifferences.central_fdm(5, 1)

gradient(f, s::Symbol, args) = gradient(f, Val(s), args)

function gradient(f, ::Val{:Zygote}, args)
    g = first(Zygote.gradient(f, args))
    if isnothing(g)
        if args isa AbstractArray{<:Real}
            return zeros(size(args)) # To respect the same output as other ADs
        else
            return zeros.(size.(args))
        end
    else
        return g
    end
end

function gradient(f, ::Val{:ForwardDiff}, args)
    ForwardDiff.gradient(f, args)
end

function gradient(f, ::Val{:ReverseDiff}, args)
    ReverseDiff.gradient(f, args)
end

function gradient(f, ::Val{:FiniteDiff}, args)
    first(FiniteDifferences.grad(FDM, f, args))
end

function compare_gradient(f, AD::Symbol, args)
    isapprox(gradient(f, AD, args), gradient(f, :FiniteDiff, args), atol=1e-8, rtol=1e-5)
end

testfunction(k, A, B, dim) = sum(kernelmatrix(k, A, B, obsdim = dim))
testfunction(k, A, dim) = sum(kernelmatrix(k, A, obsdim = dim))

function test_ADs(kernelfunction, args = nothing; ADs = [:Zygote, :ForwardDiff, :ReverseDiff], dims = [3, 3])
    test_fd = test_FiniteDiff(kernelfunction, args, dims)
    if !test_fd.anynonpass
        for AD in ADs
            test_AD(AD, kernelfunction, args, dims)
        end
    end
end

function test_FiniteDiff(kernelfunction, args = nothing, dims = [3, 3])
    # Init arguments :
    k = if args === nothing
        kernelfunction()
    else
        kernelfunction(args)
    end
    rng = MersenneTwister(42)
    @testset "FiniteDifferences" begin
        if k isa SimpleKernel
            for d in log.([eps(), rand(rng)])
                @test_nowarn gradient(:FiniteDiff, [d]) do x
                    kappa(k, exp(first(x)))
                end
            end
        end
        ## Testing Kernel Functions
        x = rand(rng, dims[1])
        y = rand(rng, dims[1])
        @test_nowarn gradient(:FiniteDiff, x) do x
                k(x, y)
            end
        if !(args === nothing)
            @test_nowarn gradient(:FiniteDiff, args) do p
                kernelfunction(p)(x, y)
            end
        end
        ## Testing Kernel Matrices
        A = rand(rng, dims...)
        B = rand(rng, dims...)
        for dim in 1:2
            @test_nowarn gradient(:FiniteDiff, A) do a
                testfunction(k, a, dim)
            end
            @test_nowarn gradient(:FiniteDiff , A) do a
                testfunction(k, a, B, dim)
            end
            @test_nowarn gradient(:FiniteDiff, B) do b
                testfunction(k, A, b, dim)
            end
            if !(args === nothing)
                @test_nowarn gradient(:FiniteDiff, args) do p
                    testfunction(kernelfunction(p), A, B, dim)
                end
            end
        end
    end
end

function test_AD(AD::Symbol, kernelfunction, args = nothing, dims = [3, 3])
    @testset "$(AD)" begin
        # Test kappa function
        k = if args === nothing
            kernelfunction()
        else
            kernelfunction(args)
        end
        rng = MersenneTwister(42)
        if k isa SimpleKernel
            for d in log.([eps(), rand(rng)])
                @test compare_gradient(AD, [d]) do x
                    kappa(k, exp(x[1])
                end
            end
        end
        # Testing kernel evaluations
        x = rand(rng, dims[1])
        y = rand(rng, dims[1])
        @test compare_gradient(AD, x) do x
            k(x, y)
        end
        @test compare_gradient(AD, y) do y
            k(x, y)
        end
        if !(args === nothing)
            @test compare_gradient(AD, args) do p
                kernelfunction(p)(x,y)
            end
        end
        # Testing kernel matrices
        A = rand(rng, dims...)
        B = rand(rng, dims...)
        for dim in 1:2
            @test compare_gradient(AD, A) do a
                testfunction(k, a, dim)
            end
            @test conpare_gradient(AD, A) do a
                testfunction(k, a, B, dim)
            end
            @test compare_gradient(AD, B) do b
                testfunction(k, A, b, dim)
            end
            if !(args === nothing)
                @test compare_gradient(AD, args) do p
                    testfunction(kernelfunction(p), AD, A, dim)
                end
            end
        end
    end
end

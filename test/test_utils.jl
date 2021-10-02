# More test utilities. Can't be included in KernelFunctions because they introduce a number
# of additional deps that we don't want to have in the main package.

# Check parameters of kernels. `trainable`, `params!`, and `params` are taken directly from
# Flux.jl so as to avoid having to depend on Flux at test-time.
trainable(m) = functor(m)[1]

params!(p::Zygote.Params, x::AbstractArray{<:Number}, seen=Zygote.IdSet()) = push!(p, x)

function params!(p::Zygote.Params, x, seen=Zygote.IdSet())
    x in seen && return nothing
    push!(seen, x)
    for child in trainable(x)
        params!(p, child, seen)
    end
end

function params(m...)
    ps = Zygote.Params()
    params!(ps, m)
    return ps
end

function test_params(kernel, reference)
    params_kernel = params(kernel)
    params_reference = params(reference)

    @test length(params_kernel) == length(params_reference)
    @test all(p == q for (p, q) in zip(params_kernel, params_reference))
end

# AD utilities


# Type to work around some performance issues that can happen on the reverse-pass of Zygote.
using Zygote: @adjoint, accum, AContext

# This context doesn't allow any globals. Don't use this if you use globals in your
# programme.
struct NoContext <: Zygote.AContext end

Zygote.cache(cx::NoContext) = (cache_fields=nothing)
Base.haskey(cx::NoContext, x) = false
Zygote.accum_param(::NoContext, x, Δ) = Δ


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
    return ForwardDiff.gradient(f, args)
end

function gradient(f, ::Val{:ReverseDiff}, args)
    return ReverseDiff.gradient(f, args)
end

function gradient(f, ::Val{:FiniteDiff}, args)
    return first(FiniteDifferences.grad(FDM, f, args))
end

function compare_gradient(f, AD::Symbol, args)
    grad_AD = gradient(f, AD, args)
    grad_FD = gradient(f, :FiniteDiff, args)
    @test grad_AD ≈ grad_FD atol = 1e-8 rtol = 1e-5
end

testfunction(k, A, B, dim) = sum(kernelmatrix(k, A, B; obsdim=dim))
testfunction(k, A, dim) = sum(kernelmatrix(k, A; obsdim=dim))
testdiagfunction(k, A, dim) = sum(kernelmatrix_diag(k, A; obsdim=dim))
testdiagfunction(k, A, B, dim) = sum(kernelmatrix_diag(k, A, B; obsdim=dim))

testfunction(k::MOKernel, A, B) = sum(kernelmatrix(k, A, B))
testfunction(k::MOKernel, A) = sum(kernelmatrix(k, A))
testdiagfunction(k::MOKernel, A) = sum(kernelmatrix_diag(k, A))
testdiagfunction(k::MOKernel, A, B) = sum(kernelmatrix_diag(k, A, B))

function test_ADs(
    kernelfunction, args=nothing; ADs=[:Zygote, :ForwardDiff, :ReverseDiff], dims=[3, 3]
)
    test_fd = test_FiniteDiff(kernelfunction, args, dims)
    if !test_fd.anynonpass
        for AD in ADs
            test_AD(AD, kernelfunction, args, dims)
        end
    end
end

function check_zygote_type_stability(f, args...; ctx=Zygote.Context())
    @inferred f(args...)
    @inferred Zygote._pullback(ctx, f, args...)
    out, pb = Zygote._pullback(ctx, f, args...)
    @inferred pb(out)
end

function test_ADs(
    k::MOKernel; ADs=[:Zygote, :ForwardDiff, :ReverseDiff], dims=(in=3, out=2, obs=3)
)
    test_fd = test_FiniteDiff(k, dims)
    if !test_fd.anynonpass
        for AD in ADs
            test_AD(AD, k, dims)
        end
    end
end

function test_FiniteDiff(kernelfunction, args=nothing, dims=[3, 3])
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
            @test_nowarn gradient(:FiniteDiff, A) do a
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

            @test_nowarn gradient(:FiniteDiff, A) do a
                testdiagfunction(k, a, dim)
            end
            @test_nowarn gradient(:FiniteDiff, A) do a
                testdiagfunction(k, a, B, dim)
            end
            @test_nowarn gradient(:FiniteDiff, B) do b
                testdiagfunction(k, A, b, dim)
            end
            if args !== nothing
                @test_nowarn gradient(:FiniteDiff, args) do p
                    testdiagfunction(kernelfunction(p), A, B, dim)
                end
            end
        end
    end
end

function test_FiniteDiff(k::MOKernel, dims=(in=3, out=2, obs=3))
    rng = MersenneTwister(42)
    @testset "FiniteDifferences" begin
        ## Testing Kernel Functions
        x = (rand(rng, dims.in), rand(rng, 1:(dims.out)))
        y = (rand(rng, dims.in), rand(rng, 1:(dims.out)))

        @test_nowarn gradient(:FiniteDiff, x[1]) do a
            k((a, x[2]), y)
        end

        ## Testing Kernel Matrices

        A = [(randn(rng, dims.in), rand(rng, 1:(dims.out))) for i in 1:(dims.obs)]
        B = [(randn(rng, dims.in), rand(rng, 1:(dims.out))) for i in 1:(dims.obs)]

        @test_nowarn gradient(:FiniteDiff, reduce(hcat, first.(A))) do a
            A = tuple.(eachcol(a), last.(A))
            testfunction(k, A)
        end
        @test_nowarn gradient(:FiniteDiff, reduce(hcat, first.(A))) do a
            A = tuple.(eachcol(a), last.(A))
            testfunction(k, A, B)
        end
        @test_nowarn gradient(:FiniteDiff, reduce(hcat, first.(B))) do b
            B = tuple.(eachcol(b), last.(B))
            testfunction(k, A, B)
        end

        @test_nowarn gradient(:FiniteDiff, reduce(hcat, first.(A))) do a
            A = tuple.(eachcol(a), last.(A))
            testdiagfunction(k, A)
        end
        @test_nowarn gradient(:FiniteDiff, reduce(hcat, first.(A))) do a
            A = tuple.(eachcol(a), last.(A))
            testdiagfunction(k, A, B)
        end
        @test_nowarn gradient(:FiniteDiff, reduce(hcat, first.(B))) do b
            B = tuple.(eachcol(b), last.(B))
            testdiagfunction(k, A, B)
        end
    end
end

function test_AD(AD::Symbol, kernelfunction, args=nothing, dims=[3, 3])
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
                compare_gradient(AD, [d]) do x
                    kappa(k, exp(x[1]))
                end
            end
        end
        # Testing kernel evaluations
        x = rand(rng, dims[1])
        y = rand(rng, dims[1])
        compare_gradient(AD, x) do x
            k(x, y)
        end
        compare_gradient(AD, y) do y
            k(x, y)
        end
        if !(args === nothing)
            compare_gradient(AD, args) do p
                kernelfunction(p)(x, y)
            end
        end
        # Testing kernel matrices
        A = rand(rng, dims...)
        B = rand(rng, dims...)
        for dim in 1:2
            compare_gradient(AD, A) do a
                testfunction(k, a, dim)
            end
            compare_gradient(AD, A) do a
                testfunction(k, a, B, dim)
            end
            compare_gradient(AD, B) do b
                testfunction(k, A, b, dim)
            end
            if !(args === nothing)
                compare_gradient(AD, args) do p
                    testfunction(kernelfunction(p), A, dim)
                end
            end

            compare_gradient(AD, A) do a
                testdiagfunction(k, a, dim)
            end
            compare_gradient(AD, A) do a
                testdiagfunction(k, a, B, dim)
            end
            compare_gradient(AD, B) do b
                testdiagfunction(k, A, b, dim)
            end
            if args !== nothing
                compare_gradient(AD, args) do p
                    testdiagfunction(kernelfunction(p), A, dim)
                end
            end
        end
    end
end

function test_AD(AD::Symbol, k::MOKernel, dims=(in=3, out=2, obs=3))
    @testset "$(AD)" begin
        rng = MersenneTwister(42)

        # Testing kernel evaluations
        x = (rand(rng, dims.in), rand(rng, 1:(dims.out)))
        y = (rand(rng, dims.in), rand(rng, 1:(dims.out)))

        compare_gradient(AD, x[1]) do a
            k((a, x[2]), y)
        end
        compare_gradient(AD, y[1]) do b
            k(x, (b, y[2]))
        end

        # Testing kernel matrices
        A = [(randn(rng, dims.in), rand(rng, 1:(dims.out))) for i in 1:(dims.obs)]
        B = [(randn(rng, dims.in), rand(rng, 1:(dims.out))) for i in 1:(dims.obs)]

        compare_gradient(AD, reduce(hcat, first.(A))) do a
            A = tuple.(eachcol(a), last.(A))
            testfunction(k, A)
        end
        compare_gradient(AD, reduce(hcat, first.(A))) do a
            A = tuple.(eachcol(a), last.(A))
            testfunction(k, A, B)
        end
        compare_gradient(AD, reduce(hcat, first.(B))) do b
            B = tuple.(eachcol(b), last.(B))
            testfunction(k, A, B)
        end
        compare_gradient(AD, reduce(hcat, first.(A))) do a
            A = tuple.(eachcol(a), last.(A))
            testdiagfunction(k, A)
        end
        compare_gradient(AD, reduce(hcat, first.(A))) do a
            A = tuple.(eachcol(a), last.(A))
            testdiagfunction(k, A, B)
        end
        compare_gradient(AD, reduce(hcat, first.(B))) do b
            B = tuple.(eachcol(b), last.(B))
            testdiagfunction(k, A, B)
        end
    end
end

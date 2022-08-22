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
# This context doesn't allow any globals. Don't use this if you use globals in your
# programme.
struct NoContext <: Zygote.AContext end

Zygote.cache(cx::NoContext) = (cache_fields = nothing)
Base.haskey(cx::NoContext, x) = false
Zygote.accum_param(::NoContext, x, Δ) = Δ

const FDM = FiniteDifferences.central_fdm(5, 1)

gradient(f, s::Symbol, args) = gradient(f, Val(s), args)

function gradient(f, ::Val{:Zygote}, args)
    g = only(Zygote.gradient(f, args))
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
    return only(FiniteDifferences.grad(FDM, f, args))
end

function compare_gradient(f, ::Val{:FiniteDiff}, args)
    @test_nowarn gradient(f, :FiniteDiff, args)
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
    test_fd = test_AD(:FiniteDiff, kernelfunction, args, dims)
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
        k = if args === nothing
            kernelfunction()
        else
            kernelfunction(args)
        end
        rng = MersenneTwister(42)

        if k isa SimpleKernel
            @testset "kappa function" begin
                for d in log.([eps(), rand(rng)])
                    compare_gradient(AD, [d]) do x
                        kappa(k, exp(x[1]))
                    end
                end
            end
        end

        @testset "kernel evaluations" begin
            x = rand(rng, dims[1])
            y = rand(rng, dims[1])
            compare_gradient(AD, x) do x
                k(x, y)
            end
            compare_gradient(AD, y) do y
                k(x, y)
            end
            if !(args === nothing)
                @testset "hyperparameters" begin
                    compare_gradient(AD, args) do p
                        kernelfunction(p)(x, y)
                    end
                end
            end
        end

        @testset "kernel matrices" begin
            A = rand(rng, dims...)
            B = rand(rng, dims...)
            @testset "$(_testfn)" for _testfn in (testfunction, testdiagfunction)
                for dim in 1:2
                    compare_gradient(AD, A) do a
                        _testfn(k, a, dim)
                    end
                    compare_gradient(AD, A) do a
                        _testfn(k, a, B, dim)
                    end
                    compare_gradient(AD, B) do b
                        _testfn(k, A, b, dim)
                    end
                    if !(args === nothing)
                        @testset "hyperparameters" begin
                            compare_gradient(AD, args) do p
                                _testfn(kernelfunction(p), A, dim)
                            end
                            compare_gradient(AD, args) do p
                                _testfn(kernelfunction(p), A, B, dim)
                            end
                        end
                    end
                end
            end
        end # kernel matrices
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

function count_allocs(f, args...)
    stats = @timed f(args...)
    return Base.gc_alloc_count(stats.gcstats)
end

"""
    constant_allocs_heuristic(f, args1::T, args2::T) where {T}

True if number of allocations associated with evaluating `f(args1...)` is equal to those
required to evaluate `f(args2...)`. Runs `f` beforehand to ensure that compilation-related
allocations are not included.
"""
function constant_allocs_heuristic(f, args1::T, args2::T) where {T}

    # Ensure that we're not counting allocations associated with compilation.
    f(args1...)
    f(args2...)

    allocs_1 = count_allocs(f, args1...)
    allocs_2 = count_allocs(f, args2...)
    return allocs_1 == allocs_2
end

"""
    ad_constant_allocs_heuristic(f, args1::T, args2::T; Δ1=nothing, Δ2=nothing) where {T}

Assesses `constant_allocs_heuristic` for `f`, `Zygote.pullback(f, args...)` and its
pullback for both of `args1` and `args2`.

`Δ1` and `Δ2` are passed to the pullback associated with `Zygote.pullback(f, args1...)`
and `Zygote.pullback(f, args2...)` respectively. If left as `nothing`, it is assumed that
the output of the primal is an acceptable cotangent to be passed to the corresponding
pullback.
"""
function ad_constant_allocs_heuristic(
    f, args1::T, args2::T; Δ1=nothing, Δ2=nothing
) where {T}
    # Check that primal has constant allocations.
    primal_heuristic = constant_allocs_heuristic(f, args1, args2)

    # Check that forwards-pass has constant allocations.
    forwards_heuristic = constant_allocs_heuristic(
        (args...) -> Zygote.pullback(f, args...), args1, args2
    )

    # Check that pullback has constant allocations for both arguments. Run twice to remove
    # compilation-related allocations.

    # First thing
    out1, pb1 = Zygote.pullback(f, args1...)
    Δ1_val = Δ1 === nothing ? out1 : Δ1
    pb1(Δ1_val)
    allocs_1 = count_allocs(pb1, Δ1_val)

    # Second thing
    out2, pb2 = Zygote.pullback(f, args2...)
    Δ2_val = Δ2 === nothing ? out2 : Δ2
    pb2(Δ2_val)
    allocs_2 = count_allocs(pb2, Δ2 === nothing ? out2 : Δ2)

    pullback_heuristic = allocs_1 == allocs_2
    return primal_heuristic, forwards_heuristic, pullback_heuristic
end

function test_zygote_perf_heuristic(f, name::String, args1, args2, Δ1=nothing, Δ2=nothing)
    @testset "$name" begin
        primal, fwd, pb = ad_constant_allocs_heuristic(f, args1, args2; Δ1, Δ2)
        @test primal
        @test fwd
        @test pb
    end
end

function test_interface_ad_perf(
    k::Kernel,
    x1::AbstractVector,
    x2::AbstractVector,
    x3::AbstractVector,
    x4::AbstractVector,
)
    test_zygote_perf_heuristic("kernelmatrix (unary)", (x1,), (x2,)) do x
        kernelmatrix(k, x)
    end
    test_zygote_perf_heuristic("kernelmatrix (binary)", (x1, x2), (x3, x4)) do x, x′
        kernelmatrix(k, x, x′)
    end
    test_zygote_perf_heuristic("kernelmatrix_diag (unary)", (x1,), (x2,)) do x
        kernelmatrix_diag(k, x)
    end
    test_zygote_perf_heuristic("kernelmatrix_diag (binary)", (x1,), (x2,)) do x
        kernelmatrix_diag(k, x, x)
    end
end

test_interface_ad_perf(k::Kernel, xs::Tuple) = test_interface_ad_perf(k, xs...)

__default_input_types() = [
    Vector{Float64},
    ColVecs{Float64, Matrix{Float64}},
    RowVecs{Float64, Matrix{Float64}},
]

function test_interface_ad_perf(k::Kernel, rng::AbstractRNG, types=__default_input_types())
    @testset "AD Alloc Performance ($T)" for T in types
        test_interface_ad_perf(k, __example_inputs(rng, T))
    end
end

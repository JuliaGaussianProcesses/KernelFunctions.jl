module TestUtils

const __ATOL = 1e-9

using LinearAlgebra
using KernelFunctions
using Random
using Test

"""
    test_interface(
        k::Kernel,
        x0::AbstractVector,
        x1::AbstractVector,
        x2::AbstractVector;
        atol=__ATOL,
    )

Run various consistency checks on `k` at the inputs `x0`, `x1`, and `x2`.
`x0` and `x1` should be of the same length with different values, while `x0` and `x2` should
be of different lengths.

    test_interface([rng::AbstractRNG], k::Kernel, T::Type{<:AbstractVector}; atol=__ATOL)

`test_interface` offers certain types of test data generation to make running these tests
require less code for common input types. For example, `Vector{<:Real}`, `ColVecs{<:Real}`,
and `RowVecs{<:Real}` are supported. For other input vector types, please provide the data
manually. 
"""
function test_interface(
    k::Kernel,
    x0::AbstractVector,
    x1::AbstractVector,
    x2::AbstractVector;
    atol=__ATOL,
)
    # TODO: uncomment the tests of ternary kerneldiagmatrix.

    # Ensure that we have the required inputs.
    @assert length(x0) == length(x1)
    @assert length(x0) ≠ length(x2)

    # Check that kerneldiagmatrix basically works.
    # @test kerneldiagmatrix(k, x0, x1) isa AbstractVector
    # @test length(kerneldiagmatrix(k, x0, x1)) == length(x0)

    # Check that pairwise basically works.
    @test kernelmatrix(k, x0, x2) isa AbstractMatrix
    @test size(kernelmatrix(k, x0, x2)) == (length(x0), length(x2))

    # Check that elementwise is consistent with pairwise.
    # @test kerneldiagmatrix(k, x0, x1) ≈ diag(kernelmatrix(k, x0, x1)) atol=atol

    # Check additional binary elementwise properties for kernels.
    # @test kerneldiagmatrix(k, x0, x1) ≈ kerneldiagmatrix(k, x1, x0)
    @test kernelmatrix(k, x0, x2) ≈ kernelmatrix(k, x2, x0)' atol=atol

    # Check that unary elementwise basically works.
    @test kerneldiagmatrix(k, x0) isa AbstractVector
    @test length(kerneldiagmatrix(k, x0)) == length(x0)

    # Check that unary pairwise basically works.
    @test kernelmatrix(k, x0) isa AbstractMatrix
    @test size(kernelmatrix(k, x0)) == (length(x0), length(x0))
    @test kernelmatrix(k, x0) ≈ kernelmatrix(k, x0)' atol=atol

    # Check that unary elementwise is consistent with unary pairwise.
    @test kerneldiagmatrix(k, x0) ≈ diag(kernelmatrix(k, x0)) atol=atol

    # Check that unary pairwise produces a positive definite matrix (approximately).
    @test all(eigvals(Matrix(kernelmatrix(k, x0))) .> -atol)

    # Check that unary elementwise / pairwise are consistent with the binary versions.
    # @test kerneldiagmatrix(k, x0) ≈ kerneldiagmatrix(k, x0, x0) atol=atol
    @test kernelmatrix(k, x0) ≈ kernelmatrix(k, x0, x0) atol=atol

    # Check that basic kernel evaluation succeeds and is consistent with `kernelmatrix`.
    @test k(first(x0), first(x1)) isa Real
    @test kernelmatrix(k, x0, x2) ≈ [k(xl, xr) for xl in x0, xr in x2]

    tmp = Matrix{Float64}(undef, length(x0), length(x2))
    @test kernelmatrix!(tmp, k, x0, x2) ≈ kernelmatrix(k, x0, x2)

    tmp_square = Matrix{Float64}(undef, length(x0), length(x0))
    @test kernelmatrix!(tmp_square, k, x0) ≈ kernelmatrix(k, x0)

    tmp_diag = Vector{Float64}(undef, length(x0))
    @test kerneldiagmatrix!(tmp_diag, k, x0) ≈ kerneldiagmatrix(k, x0)
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{Vector{T}}; kwargs...
) where {T<:Real}
    test_interface(k, randn(rng, T, 3), randn(rng, T, 3), randn(rng, T, 2); kwargs...)
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:ColVecs{T}}; dim_in=2, kwargs...,
) where {T<:Real}
    test_interface(
        k,
        ColVecs(randn(rng, T, dim_in, 3)),
        ColVecs(randn(rng, T, dim_in, 3)),
        ColVecs(randn(rng, T, dim_in, 2));
        kwargs...,
    )
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:RowVecs{T}}; dim_in=2, kwargs...,
) where {T<:Real}
    test_interface(
        k,
        RowVecs(randn(rng, T, 3, dim_in)),
        RowVecs(randn(rng, T, 3, dim_in)),
        RowVecs(randn(rng, T, 2, dim_in));
        kwargs...,
    )
end

function test_interface(k::Kernel, T::Type{<:AbstractVector}; kwargs...)
    test_interface(Random.GLOBAL_RNG, k, T; kwargs...)
end

function test_interface(rng::AbstractRNG, k::Kernel, T::Type{<:Real}; kwargs...)
    test_interface(rng, k, Vector{T}; kwargs...)
    test_interface(rng, k, ColVecs{T}; kwargs...)
    test_interface(rng, k, RowVecs{T}; kwargs...)
end

function test_interface(k::Kernel, T::Type{<:Real}; kwargs...)
    test_interface(Random.GLOBAL_RNG, k, T; kwargs...)
end

# Check parameters of kernels

function test_params(kernel, reference)
    params_kernel = params(kernel)
    params_reference = params(reference)

    @test length(params_kernel) == length(params_reference)
    @test all(p == q for (p, q) in zip(params_kernel, params_reference))
end

# AD utilities

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
    grad_AD = gradient(f, AD, args)
    grad_FD = gradient(f, :FiniteDiff, args)
    @test grad_AD ≈ grad_FD atol=1e-8 rtol=1e-5
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
                kernelfunction(p)(x,y)
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
        end
    end
end

end # module

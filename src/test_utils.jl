module TestUtils

const __ATOL = 1e-9
const __RTOL = 1e-9

using Distances
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
    rtol=__RTOL,
)
    # Ensure that we have the required inputs.
    @assert length(x0) == length(x1)
    @assert length(x0) ≠ length(x2)

    # Check that kernelmatrix_diag basically works.
    @test kernelmatrix_diag(k, x0, x1) isa AbstractVector
    @test length(kernelmatrix_diag(k, x0, x1)) == length(x0)

    # Check that pairwise basically works.
    @test kernelmatrix(k, x0, x2) isa AbstractMatrix
    @test size(kernelmatrix(k, x0, x2)) == (length(x0), length(x2))

    # Check that elementwise is consistent with pairwise.
    @test kernelmatrix_diag(k, x0, x1) ≈ diag(kernelmatrix(k, x0, x1)) atol = atol

    # Check additional binary elementwise properties for kernels.
    @test kernelmatrix_diag(k, x0, x1) ≈ kernelmatrix_diag(k, x1, x0)
    @test kernelmatrix(k, x0, x2) ≈ kernelmatrix(k, x2, x0)' atol = atol

    # Check that unary elementwise basically works.
    @test kernelmatrix_diag(k, x0) isa AbstractVector
    @test length(kernelmatrix_diag(k, x0)) == length(x0)

    # Check that unary pairwise basically works.
    @test kernelmatrix(k, x0) isa AbstractMatrix
    @test size(kernelmatrix(k, x0)) == (length(x0), length(x0))
    @test kernelmatrix(k, x0) ≈ kernelmatrix(k, x0)' atol = atol

    # Check that unary elementwise is consistent with unary pairwise.
    @test kernelmatrix_diag(k, x0) ≈ diag(kernelmatrix(k, x0)) atol = atol

    # Check that unary pairwise produces a positive definite matrix (approximately).
    @test eigmin(Matrix(kernelmatrix(k, x0))) > -atol

    # Check that unary elementwise / pairwise are consistent with the binary versions.
    @test kernelmatrix_diag(k, x0) ≈ kernelmatrix_diag(k, x0, x0) atol = atol rtol = rtol
    @test kernelmatrix(k, x0) ≈ kernelmatrix(k, x0, x0) atol = atol rtol = rtol

    # Check that basic kernel evaluation succeeds and is consistent with `kernelmatrix`.
    @test k(first(x0), first(x1)) isa Real
    @test kernelmatrix(k, x0, x2) ≈ [k(xl, xr) for xl in x0, xr in x2]

    tmp = Matrix{Float64}(undef, length(x0), length(x2))
    @test kernelmatrix!(tmp, k, x0, x2) ≈ kernelmatrix(k, x0, x2)

    tmp_square = Matrix{Float64}(undef, length(x0), length(x0))
    @test kernelmatrix!(tmp_square, k, x0) ≈ kernelmatrix(k, x0)

    tmp_diag = Vector{Float64}(undef, length(x0))
    @test kernelmatrix_diag!(tmp_diag, k, x0) ≈ kernelmatrix_diag(k, x0)
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{Vector{T}}; kwargs...
) where {T<:Real}
    return test_interface(
        k, randn(rng, T, 1001), randn(rng, T, 1001), randn(rng, T, 1000); kwargs...
    )
end

function test_interface(
    rng::AbstractRNG, k::MOKernel, ::Type{Vector{Tuple{T,Int}}}; dim_out=1, kwargs...
) where {T<:Real}
    return test_interface(
        k,
        [(randn(rng, T), rand(rng, 1:dim_out)) for i in 1:51],
        [(randn(rng, T), rand(rng, 1:dim_out)) for i in 1:51],
        [(randn(rng, T), rand(rng, 1:dim_out)) for i in 1:50];
        kwargs...,
    )
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:ColVecs{T}}; dim_in=2, kwargs...
) where {T<:Real}
    return test_interface(
        k,
        ColVecs(randn(rng, T, dim_in, 1001)),
        ColVecs(randn(rng, T, dim_in, 1001)),
        ColVecs(randn(rng, T, dim_in, 1000));
        kwargs...,
    )
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:RowVecs{T}}; dim_in=2, kwargs...
) where {T<:Real}
    return test_interface(
        k,
        RowVecs(randn(rng, T, 1001, dim_in)),
        RowVecs(randn(rng, T, 1001, dim_in)),
        RowVecs(randn(rng, T, 1000, dim_in));
        kwargs...,
    )
end

function test_interface(
    rng::AbstractRNG,
    k::Kernel,
    ::Type{<:Vector{Vector{T}}};
    dim_in=2,
    kwargs...,
) where {T<:Real}
    return test_interface(
        k,
        [randn(rng, T, dim_in) for _ in 1:1001],
        [randn(rng, T, dim_in) for _ in 1:1001],
        [randn(rng, T, dim_in) for _ in 1:1000];
        kwargs...,
    )
end

function test_interface(k::Kernel, T::Type{<:AbstractVector}; kwargs...)
    return test_interface(Random.GLOBAL_RNG, k, T; kwargs...)
end

function test_interface(rng::AbstractRNG, k::Kernel, T::Type{<:Real}; kwargs...)
    @testset "Vector{$T}" begin
        test_interface(rng, k, Vector{T}; kwargs...)
    end
    @testset "ColVecs{$T}" begin
        test_interface(rng, k, ColVecs{T}; kwargs...)
    end
    @testset "RowVecs{$T}" begin
        test_interface(rng, k, RowVecs{T}; kwargs...)
    end
    @testset "Vector{Vector{T}}" begin
        test_interface(rng, k, Vector{Vector{T}}; kwargs...)
    end
end

function test_interface(k::Kernel, T::Type{<:Real}=Float64; kwargs...)
    return test_interface(Random.GLOBAL_RNG, k, T; kwargs...)
end

end # module

module TestUtils

using CUDA
using Distances
using FillArrays
using LinearAlgebra
using KernelFunctions
using Random
using Test
using InteractiveUtils

"""
    test_interface(
        k::Kernel,
        x0::AbstractVector,
        x1::AbstractVector,
        x2::AbstractVector;
        rtol=1e-6,
        atol=rtol,
    )

Run various consistency checks on `k` at the inputs `x0`, `x1`, and `x2`.
`x0` and `x1` should be of the same length with different values, while `x0` and `x2` should
be of different lengths.

These tests are intended to pick up on really substantial issues with a kernel
implementation (e.g. substantial asymmetry in the kernel matrix, large negative
eigenvalues), rather than to test the numerics in detail, which can be kernel-specific.
"""
function test_interface(
    k::Kernel,
    x0::AbstractVector,
    x1::AbstractVector,
    x2::AbstractVector;
    rtol=1e-6,
    atol=rtol,
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
    @test kernelmatrix_diag(k, x0, x1) ≈ diag(kernelmatrix(k, x0, x1)) atol = atol rtol =
        rtol

    # Check additional binary elementwise properties for kernels.
    @test kernelmatrix_diag(k, x0, x1) ≈ kernelmatrix_diag(k, x1, x0)
    @test kernelmatrix(k, x0, x2) ≈ kernelmatrix(k, x2, x0)' atol = atol rtol = rtol

    # Check that unary elementwise basically works.
    @test kernelmatrix_diag(k, x0) isa AbstractVector
    @test length(kernelmatrix_diag(k, x0)) == length(x0)

    # Check that unary pairwise basically works.
    @test kernelmatrix(k, x0) isa AbstractMatrix
    @test size(kernelmatrix(k, x0)) == (length(x0), length(x0))
    @test kernelmatrix(k, x0) ≈ kernelmatrix(k, x0)' atol = atol rtol = rtol

    # Check that unary elementwise is consistent with unary pairwise.
    @test kernelmatrix_diag(k, x0) ≈ diag(kernelmatrix(k, x0)) atol = atol rtol = rtol

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
    @test kernelmatrix_diag!(tmp_diag, k, x0, x1) ≈ kernelmatrix_diag(k, x0, x1)
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{Vector{T}}; kwargs...
) where {T<:Real}
    return test_interface(
        k, randn(rng, T, 11), randn(rng, T, 11), randn(rng, T, 13); kwargs...
    )
end

function test_interface(
    rng::AbstractRNG, k::MOKernel, ::Type{Vector{Tuple{T,Int}}}; dim_out=3, kwargs...
) where {T<:Real}
    return test_interface(
        k,
        [(randn(rng, T), rand(rng, 1:dim_out)) for i in 1:11],
        [(randn(rng, T), rand(rng, 1:dim_out)) for i in 1:11],
        [(randn(rng, T), rand(rng, 1:dim_out)) for i in 1:13];
        kwargs...,
    )
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:ColVecs{T}}; dim_in=2, kwargs...
) where {T<:Real}
    return test_interface(
        k,
        ColVecs(randn(rng, T, dim_in, 11)),
        ColVecs(randn(rng, T, dim_in, 11)),
        ColVecs(randn(rng, T, dim_in, 13));
        kwargs...,
    )
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:ColVecs{T, CuMatrix{T}}}; dim_in=2, kwargs...
) where {T<:Real}
    return test_interface(
        k,
        _to_cuda_gpu(ColVecs(randn(rng, T, dim_in, 11))),
        _to_cuda_gpu(ColVecs(randn(rng, T, dim_in, 11))),
        _to_cuda_gpu(ColVecs(randn(rng, T, dim_in, 13)));
        kwargs...,
    )
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:RowVecs{T}}; dim_in=2, kwargs...
) where {T<:Real}
    return test_interface(
        k,
        RowVecs(randn(rng, T, 11, dim_in)),
        RowVecs(randn(rng, T, 11, dim_in)),
        RowVecs(randn(rng, T, 13, dim_in));
        kwargs...,
    )
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:Vector{Vector{T}}}; dim_in=2, kwargs...
) where {T<:Real}
    return test_interface(
        k,
        [randn(rng, T, dim_in) for _ in 1:11],
        [randn(rng, T, dim_in) for _ in 1:11],
        [randn(rng, T, dim_in) for _ in 1:13];
        kwargs...,
    )
end

function test_interface(k::Kernel, T::Type{<:AbstractVector}; kwargs...)
    return test_interface(Random.GLOBAL_RNG, k, T; kwargs...)
end

"""
    test_interface([rng::AbstractRNG], k::Kernel, ::Type{T}; kwargs...) where {T<:Real}

Run the [`test_interface`](@ref) tests for randomly generated inputs of types `Vector{T}`,
`Vector{Vector{T}}`, `ColVecs{T}`, and `RowVecs{T}`.

For other input types, please provide the data manually.

The keyword arguments are forwarded to the invocations of [`test_interface`](@ref) with the
randomly generated inputs.
"""
function test_interface(rng::AbstractRNG, k::Kernel, ::Type{T}; kwargs...) where {T<:Real}
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

const FloatType = Union{Float32, Float64}

"""
    example_inputs(rng::AbstractRNG, type)

Return a tuple of 4 inputs of type `type`. See `methods(example_inputs)` for information
around supported types. It is recommended that you utilise `StableRNGs.jl` for `rng` here
to ensure consistency across Julia versions.
"""
function example_inputs(rng::AbstractRNG, ::Type{Vector{T}}) where {T<:FloatType}
    return map(n -> randn(rng, T, n), (1, 2, 3, 4))
end

function example_inputs(
    rng::AbstractRNG, ::Type{ColVecs{T,Matrix{T}}}; dim::Int=2
) where {T<:FloatType}
    return map(n -> ColVecs(randn(rng, T, dim, n)), (1, 2, 3, 4))
end

function example_inputs(
    rng::AbstractRNG, ::Type{RowVecs{T,Matrix{T}}}; dim::Int=2
) where {T<:FloatType}
    return map(n -> RowVecs(randn(rng, T, n, dim)), (1, 2, 3, 4))
end

function example_inputs(rng::AbstractRNG, ::Type{CuVector{T}}) where {T<:FloatType}
    return map(_to_cuda_gpu, example_inputs(rng, Vector{T}))
end

function example_inputs(
    rng::AbstractRNG, ::Type{ColVecs{T, CuMatrix{T}}}; dim::Int=2
) where {T<:FloatType}
    return map(_to_cuda_gpu, example_inputs(rng, ColVecs{T, Matrix{T}}; dim=dim))
end

function example_inputs(
    rng::AbstractRNG, ::Type{RowVecs{T, CuMatrix{T}}}; dim::Int=2
) where {T<:FloatType}
    return map(_to_cuda_gpu, example_inputs(rng, RowVecs{T, Matrix{T}}; dim=dim))
end

function test_gpu_against_cpu(k::Kernel, x1::AbstractVector, x2::AbstractVector; atol=1e-6)
    @assert length(x1) != length(x2)

    k_cpu = _to_cpu(k)
    x1_cpu = _to_cpu(x1)
    x2_cpu = _to_cpu(x2)
    let
        K_cpu = kernelmatrix(k_cpu, x1_cpu)
        K_gpu = kernelmatrix(k, x1)
        @test size(K_cpu) == size(K_gpu)
        @test eltype(K_cpu) == eltype(K_gpu)
        @test isapprox(K_cpu, _to_cpu(K_gpu); atol=atol)
    end
    let
        K_cpu = kernelmatrix(k_cpu, x1_cpu, x2_cpu)
        K_gpu = kernelmatrix(k, x1, x2)
        @test size(K_cpu) == size(K_gpu)
        @test eltype(K_cpu) == eltype(K_gpu)
        @test isapprox(K_cpu, _to_cpu(K_gpu); atol=atol)
    end
    let
        K_cpu = kernelmatrix_diag(k_cpu, x1_cpu)
        K_gpu = kernelmatrix_diag(k, x1)
        @test size(K_cpu) == size(K_gpu)
        @test eltype(K_cpu) == eltype(K_gpu)
        @test isapprox(K_cpu, _to_cpu(K_gpu); atol=atol)
    end
    let
        K_cpu = kernelmatrix_diag(k_cpu, x1_cpu, x1_cpu)
        K_gpu = kernelmatrix_diag(k, x1, x1)
        @test size(K_cpu) == size(K_gpu)
        @test eltype(K_cpu) == eltype(K_gpu)
        @test isapprox(K_cpu, _to_cpu(K_gpu); atol=atol)
    end
end

function test_gpu_against_cpu(rng::AbstractRNG, k::Kernel, data_type::Type)
    _, x1, x2, _ = example_inputs(rng, data_type)
    test_gpu_against_cpu(k, x1, x2)
end

_to_cpu(x::CuArray{<:Real}) = Array(x)
_to_cpu(x::ColVecs{<:Real, <:CuMatrix}) = ColVecs(_to_cpu(x.X))
_to_cpu(x::RowVecs{<:Real, <:CuMatrix}) = RowVecs(_to_cpu(x.X))
_to_cpu(x::FillArrays.AbstractFill{<:Real}) = x

_to_cpu(k::Kernel) = k
_to_cpu(k::TransformedKernel) = TransformedKernel(_to_cpu(k.kernel), _to_cpu(k.transform))
_to_cpu(t::Transform) = t
_to_cpu(t::LinearTransform) = LinearTransform(_to_cpu(t.A))
_to_cpu(t::ChainTransform) = ChainTransform(map(_to_cpu, t.transforms))

_to_cuda_gpu(x::Array{<:Real}) = CuArray(x)
_to_cuda_gpu(x::ColVecs{T, Matrix{T}} where {T<:Real}) = ColVecs(_to_cuda_gpu(x.X))
_to_cuda_gpu(x::RowVecs{T, Matrix{T}} where {T<:Real}) = RowVecs(_to_cuda_gpu(x.X))
_to_cuda_gpu(x::FillArrays.AbstractFill{<:Real}) = x

end # module

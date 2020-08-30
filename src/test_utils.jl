module TestUtils

const __ATOL = 1e-9

using KernelFunctions

"""
    test_interface(k::Kernel, x0::AV, x1::AV, x2::AV; atol=__ATOL)

Run various consistency checks on `k` at the inputs `x0`, `x1`, and `x2`.
`x0` and `x1` should be of the same length with different values, while `x0` and `x2` should
be of different lengths.

    test_interface([rng::AbstractRNG], k::Kernel, T::Type{<:AbstractVector}; atol=__ATOL)

`test_interface` offers certain types of test data generation to make running these tests
require less code for common input types. For example, `Vector{<:Real}`, `ColVecs{<:Real}`,
and `RowVecs{<:Real}` are supported. For other input vector types, please provide the data
manually. 
"""
function test_interface(k::Kernel, x0::AV, x1::AV, x2::AV; atol=__ATOL)

    # TODO: uncomment the tests of ternary kerneldiagmatrix.

    # Ensure that we have the required inputs.
    @assert length(x0) == length(x1)
    @assert length(x0) ≠ length(x2)
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
    @test kernelmatrix(k, x0, x2) ≈ [k(xl, xr) for xl in x0, xr in x2]
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{Vector{T}}; atol=__ATOL,
) where {T<:Real}
    test_interface(k, randn(rng, T, 3), randn(rng, T, 3), randn(rng, T, 2);  atol=atol)
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:ColVecs{T}}; atol=__ATOL,
) where {T<:Real}
    test_interface(
        k,
        ColVecs(randn(rng, T, 2, 3)),
        ColVecs(randn(rng, T, 2, 3)),
        ColVecs(randn(rng, T, 2, 2));
        atol=atol,
    )
end

function test_interface(
    rng::AbstractRNG, k::Kernel, ::Type{<:RowVecs{T}}; atol=__ATOL,
) where {T<:Real}
    test_interface(
        k,
        RowVecs(randn(rng, T, 3, 2)),
        RowVecs(randn(rng, T, 3, 2)),
        RowVecs(randn(rng, T, 2, 2));
        atol=atol,
    )
end

function test_interface(k::Kernel, T::Type{<:AbstractVector}; atol=__ATOL)
    test_interface(Random.GLOBAL_RNG, k, T)
end

function test_interface(rng::AbstractRNG, k::Kernel, T::Type{<:Real}; atol=__ATOL)
    test_interface(rng, k, Vector{T}; atol=atol)
    test_interface(rng, k, ColVecs{T} atol=atol)
    test_interface(rng, k, RowVecs{T} atol=atol)
end

function test_interface(k::Kernel, T::Type{<:Real}; atol=__ATOL)
    test_interface(Random.GLOBAL_RNG, k, T; atol=atol)
end

end # module

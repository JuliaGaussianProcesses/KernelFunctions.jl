# Following the algorithm by William and Seeger, 2001
# Cs is equivalent to X_mm and C to X_mn

function sampleindex(X::AbstractVector, r::Real)
    0 < r <= 1 || throw(ArgumentError("Sample rate `r` must be in range (0,1]"))
    n = length(X)
    m = ceil(Int, n * r)
    S = StatsBase.sample(1:n, m; replace=false, ordered=true)
    return S
end

function sampleindex(X::AbstractMatrix, r::Real; obsdim::Integer=defaultobs)
    return sampleindex(vec_of_vecs(X; obsdim=obsdim), r)
end

function nystrom_sample(k::Kernel, X::AbstractVector, S::Vector{<:Integer})
    Xₘ = X[S]
    C = kernelmatrix(k, Xₘ, X)
    Cs = C[:, S]
    return (C, Cs)
end

function nystrom_sample(
    k::Kernel, X::AbstractMatrix, S::Vector{<:Integer}; obsdim::Integer=defaultobs
)
    return nystrom_sample(k, vec_of_vecs(X; obsdim=obsdim), S)
end

function nystrom_pinv!(Cs::Matrix{T}, tol::T=eps(T) * size(Cs, 1)) where {T<:Real}
    # Compute eigendecomposition of sampled component of K
    QΛQᵀ = LinearAlgebra.eigen!(LinearAlgebra.Symmetric(Cs))

    # Solve for D = Λ^(-1/2) (pseudo inverse - use tolerance from before factorization)
    D = QΛQᵀ.values
    λ_tol = maximum(D) * tol

    for i in eachindex(D)
        @inbounds D[i] = abs(D[i]) <= λ_tol ? zero(T) : one(T) / sqrt(D[i])
    end

    # Scale eigenvectors by D
    Q = QΛQᵀ.vectors
    QD = LinearAlgebra.rmul!(Q, LinearAlgebra.Diagonal(D))  # Scales column i of Q by D[i]

    # W := (QD)(QD)ᵀ = (QΛQᵀ)^(-1)  (pseudo inverse)
    W = QD * QD'

    # Symmetrize W
    return LinearAlgebra.copytri!(W, 'U')
end

@doc raw"""
    NystromFact

Type for storing a Nystrom factorization. The factorization contains two fields: `W` and
`C`, two matrices satisfying:
```math
\mathbf{K} \approx \mathbf{C}^{\intercal}\mathbf{W}\mathbf{C}
```
"""
struct NystromFact{T<:Real}
    W::Matrix{T}
    C::Matrix{T}
end

function NystromFact(W::Matrix{<:Real}, C::Matrix{<:Real})
    T = Base.promote_eltypeof(W, C)
    return NystromFact(convert(Matrix{T}, W), convert(Matrix{T}, C))
end

@doc raw"""
    nystrom(k::Kernel, X::Vector, S::Vector)

Computes a factorization of Nystrom approximation of the square kernel matrix
of data vector `X` with respect to kernel `k`, using indices `S`.
Returns a `NystromFact` struct which stores a Nystrom factorization satisfying:
```math
\mathbf{K} \approx \mathbf{C}^{\intercal}\mathbf{W}\mathbf{C}
```
"""
function nystrom(k::Kernel, X::AbstractVector, S::AbstractVector{<:Integer})
    C, Cs = nystrom_sample(k, X, S)
    W = nystrom_pinv!(Cs)
    return NystromFact(W, C)
end

@doc raw"""
    nystrom(k::Kernel, X::Vector, r::Real)

Computes a factorization of Nystrom approximation of the square kernel matrix
of data vector `X` with respect to kernel `k` using a sample ratio of `r`.
Returns a `NystromFact` struct which stores a Nystrom factorization satisfying:
```math
\mathbf{K} \approx \mathbf{C}^{\intercal}\mathbf{W}\mathbf{C}
```
"""
function nystrom(k::Kernel, X::AbstractVector, r::Real)
    S = sampleindex(X, r)
    return nystrom(k, X, S)
end

function nystrom(k::Kernel, X::AbstractMatrix, S::Vector{<:Integer}; obsdim::Int=defaultobs)
    return nystrom(k, vec_of_vecs(X; obsdim=obsdim), S)
end

function nystrom(k::Kernel, X::AbstractMatrix, r::Real; obsdim::Int=defaultobs)
    return nystrom(k, vec_of_vecs(X; obsdim=obsdim), r)
end

"""
    kernelmatrix(CᵀWC::NystromFact)

Compute the approximate kernel matrix based on the Nystrom factorization.
"""
function kernelmatrix(CᵀWC::NystromFact{<:Real})
    W = CᵀWC.W
    C = CᵀWC.C
    return C' * W * C
end

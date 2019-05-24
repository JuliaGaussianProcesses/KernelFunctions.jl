"""
```
    kernelmatrix!(K::Matrix, κ::Kernel, X::Matrix, Y::Matrix; obsdim::Integer=2)
```
In-place version of `kernelmatrix` where pre-allocated matrix `K` will be overwritten with the kernel matrix.
"""
function kernelmatrix!(
        K::Matrix{T},
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T};
        obsdim::Integer = defaultobs
    ) where {T<:Real}
    basematrix!(σ, K, basefunction(κ), κ.α, X, Y)
    kappamatrix!(κ, K)
end

function kernelmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T};
        obsdim::Int = defaultobs,
        symmetrize::Bool = true
    ) where {T<:Real}
    return symmetric_kappamatrix!(κ,pairwise(basefunction(κ),X,dims=obsdim),symmetrize)
end

function kernelmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T},
        Y::AbstractMatrix{T};
        obsdim::Int = defaultobs
    ) where {T<:Real}
    kappamatrix!(κ, pairwise(basefunction(κ), X, Y, dims=obsdim))
end


# Convenience Methods ======================================================================

"""
    kernel(κ::Kernel, x, y)

Apply the kernel `κ` to ``x`` and ``y`` where ``x`` and ``y`` are vectors or scalars of
some subtype of ``Real``.
"""
function kernel(κ::Kernel{T}, x::Real, y::Real) where {T}
    kernel(κ, T(x), T(y))
end

function kernel(
        κ::Kernel{T},
        x::AbstractArray{T1},
        y::AbstractArray{T2};
        obsdim::Int = defaultobs
    ) where {T,T1<:Real,T2<:Real}
    kappamatrix!(κ, pairwise(metric(κ),X,Y,dims=obsdim))
end

"""
```
    kernelmatrix(κ::Kernel, X::Matrix ; obsdim::Int=2, symmetrize::Bool)
```
Calculate the kernel matrix of `X` with respect to kernel `κ`.
"""
function kernelmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T1};
        obsdim::Int = defaultobs,
        symmetrize::Bool = true
    ) where {T,T1}
    return symmetric_kappamatrix!(κ,pairwise(basefunction(κ),X,dims=obsdim),symmetrize)
end

"""
    kernelmatrix(κ::Kernel, X::Matrix, Y::Matrix; obsdim::Int=2)

Calculate the base matrix of `X` and `Y` with respect to kernel `κ`.
"""
function kernelmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T1},
        Y::AbstractMatrix{T2};
        obsdim=defaultobs
    ) where {T,T1,T2}
    kappamatrix!(κ, pairwise(basefunction(κ), X, Y, dims=dim(σ)))
end

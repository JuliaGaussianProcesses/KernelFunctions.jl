
function _kappamatrix!(κ::Kernel{T}, P::AbstractMatrix{T₁}) where {T<:Real,T₁<:Real}
    for i in eachindex(P)
        @inbounds P[i] = kappa(κ, P[i])
    end
    P
end

function _symmetric_kappamatrix!(
        κ::Kernel{T},
        P::AbstractMatrix{T₁},
        symmetrize::Bool
    ) where {T<:Real,T₁<:Real}
    if !((n = size(P,1)) == size(P,2))
        throw(DimensionMismatch("Pairwise matrix must be square."))
    end
    for j = 1:n, i = (1:j)
        @inbounds P[i,j] = kappa(κ, P[i,j])
    end
    symmetrize ? LinearAlgebra.copytri!(P, 'U') : P
end


"""
```
    kernelmatrix!(K::Matrix, κ::Kernel, X::Matrix, Y::Matrix; obsdim::Integer=2)
```
In-place version of `kernelmatrix` where pre-allocated matrix `K` will be overwritten with the kernel matrix.
"""
function kernelmatrix!(
        K::Matrix{T₁},
        κ::Kernel{T},
        X::AbstractMatrix{T₂},
        Y::AbstractMatrix{T₃};
        obsdim::Int = defaultobs
        ) where {T,T₁,T₂,T₃}
        #TODO Check dimension consistency
        _kappamatrix!(κ, pairwise!(K,metric(κ), X, Y, dims=obsdim))
end


function kernelmatrix!(
        K::Matrix{T₁},
        κ::Kernel{T},
        X::AbstractMatrix{T₂};
        obsdim::Int = defaultobs,
        symmetrize::Bool = true
        ) where {T,T₁<:Real,T₂<:Real}
        #TODO Check dimension consistency
        _symmetric_kappamatrix!(κ,pairwise!(K,metric(κ),X,dims=obsdim),symmetrize)
end

# Convenience Methods ======================================================================

"""
```
    kernel(κ::Kernel, x, y; obsdim=2)
```
Apply the kernel `κ` to ``x`` and ``y`` where ``x`` and ``y`` are vectors or scalars of
some subtype of ``Real``.
"""
function kernel(κ::Kernel{T}, x::Real, y::Real) where {T}
    kernel(κ, T(x), T(y))
end

function kernel(
        κ::Kernel{T},
        x::AbstractArray{T₁},
        y::AbstractArray{T₂};
        obsdim::Int = defaultobs
    ) where {T,T₁<:Real,T₂<:Real}
    # TODO Verify dimensions
    _kappamatrix!(κ, pairwise(metric(κ),X,Y,dims=obsdim))
end

"""
```
    kernelmatrix(κ::Kernel, X::Matrix ; obsdim::Int=2, symmetrize::Bool)
```
Calculate the kernel matrix of `X` with respect to kernel `κ`.
"""
function kernelmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T₁};
        obsdim::Int = defaultobs,
        symmetrize::Bool = true
    ) where {T,T₁<:Real}
    return _symmetric_kappamatrix!(κ,pairwise(metric(κ),X,dims=obsdim),symmetrize)
end

"""
```
    kernelmatrix(κ::Kernel, X::Matrix, Y::Matrix; obsdim::Int=2)
```
Calculate the base matrix of `X` and `Y` with respect to kernel `κ`.
"""
function kernelmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T₁},
        Y::AbstractMatrix{T₂};
        obsdim=defaultobs
    ) where {T,T₁<:Real,T₂<:Real}
    _kappamatrix!(κ, pairwise(metric(κ), X, Y, dims=obsdim))
end


"""
```
    kerneldiagmatrix(κ::Kernel, X::Matrix; obsdim::Int=2)
```
Calculate the diagonal matrix of `X` with respect to kernel `κ`
"""
function kerneldiagmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T₁}
        ) where {T,T₁,T₂}
        @error "Not implemented yet"
        #TODO
end

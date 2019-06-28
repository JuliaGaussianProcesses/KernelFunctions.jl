
function _kappamatrix!(κ::Kernel, P::AbstractMatrix{T₁}) where {T₁<:Real}
    for i in eachindex(P)
        @inbounds P[i] = kappa(κ, P[i])
    end
    P
end

function _symmetric_kappamatrix!(
        κ::Kernel,
        P::AbstractMatrix{T₁},
        symmetrize::Bool
    ) where {T₁<:Real}
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
        K::AbstractMatrix{T₁},
        κ::Kernel{T},
        X::AbstractMatrix{T₂},
        Y::AbstractMatrix{T₃};
        obsdim::Int = defaultobs
        ) where {T,T₁,T₂,T₃}
        #TODO Check dimension consistency
        _kappamatrix!(κ, pairwise!(K, metric(κ), X, Y, dims=obsdim))
end

"""
```
    kernelmatrix!(K::Matrix, κ::Kernel, X::Matrix; obsdim::Integer=2, symmetrize::Bool=true)
```
In-place version of `kernelmatrix` where pre-allocated matrix `K` will be overwritten with the kernel matrix.
"""
function kernelmatrix!(
        K::Matrix{T₁},
        κ::Kernel{T},
        X::AbstractMatrix{T₂};
        obsdim::Int = defaultobs,
        symmetrize::Bool = true
        ) where {T,T₁<:Real,T₂<:Real}
        #TODO Check dimension consistency
        _symmetric_kappamatrix!(κ,pairwise!(K, metric(κ), X, dims=obsdim), symmetrize)
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
    kappa(κ, evaluate(metric(κ),transform(κ,x),transform(κ,y)))
end

"""
```
    kernelmatrix(κ::Kernel, X::Matrix ; obsdim::Int=2, symmetrize::Bool=true)
```
Calculate the kernel matrix of `X` with respect to kernel `κ`.
"""
function kernelmatrix(
        κ::Kernel{T,<:Transform{A}},
        X::AbstractMatrix;
        obsdim::Int = defaultobs,
        symmetrize::Bool = true
    ) where {T,A}
    # Tₖ = typeof(zero(eltype(X))*zero(T))
    # m = size(X,obsdim)
    K = map(x->kappa(κ,x),pairwise(metric(κ),transform(κ,X,obsdim),dims=obsdim))
    # K = Matrix{Tₖ}(undef,m,m)
    # for i in 1:m
    #     tx = transform(κ,@view X[i,:])
    #     for j in 1:i
    #         K[i,j] = kappa(κ,kernel(κ,tx,transform(@view X[j,:])))
    #     end
    # end
    return K
    # return kernelmatrix!(Matrix{Tₖ}(undef,m,m),κ,X,obsdim=obsdim,symmetrize=symmetrize)
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
    # Tₖ = typeof(zero(eltype(X))*zero(T))
    # m = size(X,obsdim)
    K = map(x->kappa(κ,x),pairwise(metric(κ),transform(κ,X,obsdim),transform(κ,Y,obsdim),dims=obsdim))
    # K = Matrix{Tₖ}(undef,m,m)
    # for i in 1:m
    #     tx = transform(κ,@view X[i,:])
    #     for j in 1:i
    #         K[i,j] = kappa(κ,kernel(κ,tx,transform(@view X[j,:])))
    #     end
    # end
    return K
    # return kernelmatrix!(Matrix{Tₖ}(undef,m,m),κ,X,obsdim=obsdim,symmetrize=symmetrize)
end


"""
```
    kerneldiagmatrix(κ::Kernel, X::Matrix; obsdim::Int=2)
```
Calculate the diagonal matrix of `X` with respect to kernel `κ`
"""
function kerneldiagmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T₁};
        obsdim::Int = 2
        ) where {T,T₁}
        n = size(X,obsdim)
        Tₖ = typeof(zero(T)*zero(eltype(X)))
        K = Vector{Tₖ}(undef,n)
        kerneldiagmatrix!(K,κ,X,obsdim=obsdim)
        return K
end

function kerneldiagmatrix!(
        K::AbstractVector{T₁},
        κ::Kernel{T},
        X::AbstractMatrix{T₂};
        obsdim::Int = 2
        ) where {T,T₁,T₂}
        if obsdim == 1
            for i in eachindex(K)
                @inbounds @views K[i] = kernel(κ, X[i,:],X[i,:])
            end
        else
            for i in eachindex(K)
                @inbounds @views K[i] = kernel(κ,X[:,i],X[:,i])
            end
        end
        return K
end

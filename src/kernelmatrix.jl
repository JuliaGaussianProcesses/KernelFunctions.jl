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
        obsdim::Int = defaultobs
        ) where {T,T₁<:Real,T₂<:Real}
        if !check_dims(K,X,X,feature_dim(obsdim),obsdim)
            throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
        end
        map!(x->kappa(κ,x),K,pairwise(metric(κ),transform(κ,X,obsdim),dims=obsdim))
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
        if !check_dims(K,X,Y,feature_dim(obsdim),obsdim)
            throw(DimensionMismatch("Dimensions $(size(K)) of the target array K are not consistent with X ($(size(X))) and Y ($(size(Y)))"))
        end
        map!(x->kappa(κ,x),K,pairwise(metric(κ),transform(κ,X,obsdim),transform(κ,Y,obsdim),dims=obsdim))
end

"""
```
    kernel(κ::Kernel, x, y; obsdim=2)
```
Apply the kernel `κ` to `x` and `y`.
"""
function kernel(κ::Kernel{T}, x::Real, y::Real) where {T}
    kernel(κ, [T(x)], [T(y)])
end

function kernel(
        κ::Kernel{T},
        x::AbstractArray{T₁},
        y::AbstractArray{T₂};
        obsdim::Int = defaultobs
    ) where {T,T₁<:Real,T₂<:Real}
    @assert length(x) == length(y) "x and y don't have the same dimension!"
    kappa(κ, evaluate(metric(κ),transform(κ,x),transform(κ,y)))
end

"""
```
    kernelmatrix(κ::Kernel, X::Matrix ; obsdim::Int=2, symmetrize::Bool=true)
```
Calculate the kernel matrix of `X` with respect to kernel `κ`.
`obsdim=1` means the matrix `X` has size #samples x #dimension
`obsdim=2` means the matrix `X` has size #dimension x #samples
"""
function kernelmatrix(
        κ::Kernel{T,<:Transform},
        X::AbstractMatrix;
        obsdim::Int = defaultobs,
        symmetrize::Bool = true
    ) where {T}
    K = map(x->kappa(κ,x),pairwise(metric(κ),transform(κ,X,obsdim),dims=obsdim))
end

"""
```
    kernelmatrix(κ::Kernel, X::Matrix, Y::Matrix; obsdim::Int=2)
```
Calculate the base matrix of `X` and `Y` with respect to kernel `κ`.
`obsdim=1` means the matrices `X` and `Y` have sizes #samples x #dimension
`obsdim=2` means the matrices `X` and `Y` have size #dimension x #samples
"""
function kernelmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T₁},
        Y::AbstractMatrix{T₂};
        obsdim=defaultobs
    ) where {T,T₁<:Real,T₂<:Real}
    if !check_dims(X,Y,feature_dim(obsdim),obsdim)
        throw(DimensionMismatch("X ($(size(X))) and Y ($(size(Y))) do not have the same number of features on the dimension obsdim : $(feature_dim(obsdim))"))
    end
    K = map(x->kappa(κ,x),pairwise(metric(κ),transform(κ,X,obsdim),transform(κ,Y,obsdim),dims=obsdim))
    return K
end

"""
```
    kerneldiagmatrix(κ::Kernel, X::Matrix; obsdim::Int=2)
```
Calculate the diagonal matrix of `X` with respect to kernel `κ`
`obsdim=1` means the matrix `X` has size #samples x #dimension
`obsdim=2` means the matrix `X` has size #dimension x #samples
"""
function kerneldiagmatrix(
        κ::Kernel{T},
        X::AbstractMatrix{T₁};
        obsdim::Int = defaultobs
        ) where {T,T₁}
        if obsdim == 1
            [@views kernel(κ,X[i,:],X[i,:]) for i in 1:size(X,obsdim)]
        elseif obsdim == 2
            [@views kernel(κ,X[i,:],X[i,:]) for i in 1:size(X,obsdim)]
        end
end

function kerneldiagmatrix!(
        K::AbstractVector{T₁},
        κ::Kernel{T},
        X::AbstractMatrix{T₂};
        obsdim::Int = defaultobs
        ) where {T,T₁,T₂}
        if length(K) != size(X,obsdim)
            throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
        end
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

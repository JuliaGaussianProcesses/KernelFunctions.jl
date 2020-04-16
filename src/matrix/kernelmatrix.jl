"""
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X; obsdim::Integer = 2)
    kernelmatrix!(K::AbstractMatrix, κ::Kernel, X, Y; obsdim::Integer = 2)

In-place version of [`kernelmatrix`](@ref) where pre-allocated matrix `K` will be overwritten with the kernel matrix.
"""
kernelmatrix!

function kernelmatrix!(
    K::AbstractMatrix,
    kernel::Kernel,
    X::AbstractVector;
    obsdim::Int = defaultobs
)
    return kernelmatrix!(K, kernel, reshape(X, 1, :); obsdim = 2)
end

function kernelmatrix!(
        K::AbstractMatrix,
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
        @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
        if !check_dims(K,X,X,feature_dim(obsdim),obsdim)
            throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
        end
        map!(x->kappa(κ,x),K,pairwise(metric(κ),X,dims=obsdim))
end

kernelmatrix!(K::AbstractMatrix, κ::TransformedKernel, X::AbstractMatrix; obsdim::Int = defaultobs) =
        kernelmatrix!(K, kernel(κ), apply(κ.transform, X, obsdim = obsdim), obsdim = obsdim)

function kernelmatrix!(
    K::AbstractMatrix,
    kernel::Kernel,
    X::AbstractVector,
    Y::AbstractVector;
    obsdim::Int = defaultobs
)
    return kernelmatrix!(K, kernel, reshape(X, 1, :), reshape(Y, 1, :); obsdim = 2)
end

function kernelmatrix!(
        K::AbstractMatrix,
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix;
        obsdim::Int = defaultobs
        )
        @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
        if !check_dims(K,X,Y,feature_dim(obsdim),obsdim)
            throw(DimensionMismatch("Dimensions $(size(K)) of the target array K are not consistent with X ($(size(X))) and Y ($(size(Y)))"))
        end
        map!(x->kappa(κ,x),K,pairwise(metric(κ),X,Y,dims=obsdim))
end

kernelmatrix!(K::AbstractMatrix, κ::TransformedKernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim::Int = defaultobs) =
        kernelmatrix!(K, kernel(κ), apply(κ.transform, X, obsdim = obsdim), apply(κ.transform, Y, obsdim = obsdim), obsdim = obsdim)

## Apply kernel on two reals ##
function _kernel(κ::Kernel, x::Real, y::Real)
    _kernel(κ, [x], [y])
end

## Apply kernel on two vectors ##
function _kernel(
        κ::Kernel,
        x::AbstractVector,
        y::AbstractVector;
        obsdim::Int = defaultobs
    )
    @assert length(x) == length(y) "x and y don't have the same dimension!"
    kappa(κ, evaluate(metric(κ),x,y))
end

_kernel(κ::TransformedKernel, x::AbstractVector, y::AbstractVector; obsdim::Int = defaultobs) =
        _kernel(kernel(κ), apply(κ.transform, x), apply(κ.transform, y), obsdim = obsdim)

"""
    kernelmatrix(κ::Kernel, X; obsdim::Int = 2)
    kernelmatrix(κ::Kernel, X, Y; obsdim::Int = 2)

Calculate the kernel matrix of `X` (and `Y`) with respect to kernel `κ`.
`obsdim = 1` means the matrix `X` (and `Y`) has size #samples x #dimension
`obsdim = 2` means the matrix `X` (and `Y`) has size #dimension x #samples
"""
kernelmatrix

function kernelmatrix(
        κ::Kernel,
        X::AbstractVector{<:Real};
        obsdim::Int=defaultobs
        )
        kernelmatrix(κ,reshape(X,1,:),obsdim=2)
end

function kernelmatrix(
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
        K = map(x->kappa(κ,x),pairwise(metric(κ),X,dims=obsdim))
end

kernelmatrix(κ::TransformedKernel, X::AbstractMatrix; obsdim::Int = defaultobs) =
        kernelmatrix(kernel(κ), apply(κ.transform, X, obsdim = obsdim), obsdim = obsdim)

function kernelmatrix(
    kernel::Kernel,
    X::AbstractVector{<:Real},
    Y::AbstractVector{<:Real};
    obsdim::Int = defaultobs
)
    return kernelmatrix(kernel, reshape(X, 1, :), reshape(Y, 1, :); obsdim = 2)
end

function kernelmatrix(
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix;
        obsdim=defaultobs
    )
    @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
    if !check_dims(X,Y,feature_dim(obsdim),obsdim)
        throw(DimensionMismatch("X $(size(X)) and Y $(size(Y)) do not have the same number of features on the dimension : $(feature_dim(obsdim))"))
    end
    _kernelmatrix(κ,X,Y,obsdim)
end

@inline _kernelmatrix(κ::Kernel,X,Y,obsdim) = map(x->kappa(κ,x),pairwise(metric(κ),X,Y,dims=obsdim))

kernelmatrix(κ::TransformedKernel, X::AbstractMatrix, Y::AbstractMatrix; obsdim::Int = defaultobs) =
        kernelmatrix(kernel(κ), apply(κ.transform, X, obsdim = obsdim), apply(κ.transform, Y, obsdim = obsdim), obsdim = obsdim)

"""
    kerneldiagmatrix(κ::Kernel, X; obsdim::Int = 2)

Calculate the diagonal matrix of `X` with respect to kernel `κ`
`obsdim = 1` means the matrix `X` has size #samples x #dimension
`obsdim = 2` means the matrix `X` has size #dimension x #samples
"""
function kerneldiagmatrix(
    kernel::Kernel,
    X::AbstractVector;
    obsdim::Int = defaultobs
)
    return kerneldiagmatrix(kernel, reshape(X, 1, :); obsdim = 2)
end

function kerneldiagmatrix(
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
        @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
        if obsdim == 1
            @compat eachrow(X) .|> x->_kernel(κ,x,x) #[@views _kernel(κ,X[i,:],X[i,:]) for i in 1:size(X,obsdim)]
        elseif obsdim == 2
            @compat eachcol(X) .|> x->_kernel(κ,x,x) #[@views _kernel(κ,X[:,i],X[:,i]) for i in 1:size(X,obsdim)]
        end
end

"""
    kerneldiagmatrix!(K::AbstractVector, κ::Kernel, X; obsdim::Int = 2)

In place version of [`kerneldiagmatrix`](@ref)
"""
function kerneldiagmatrix!(
    K::AbstractVector,
    kernel::Kernel,
    X::AbstractVector;
    obsdim::Int = defaultobs
)
    return kerneldiagmatrix!(K, kernel, reshape(X, 1, :); obsdim = 2)
end

function kerneldiagmatrix!(
        K::AbstractVector,
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
        @assert obsdim ∈ [1,2] "obsdim should be 1 or 2 (see docs of kernelmatrix))"
        if length(K) != size(X,obsdim)
            throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
        end
        if obsdim == 1
            for i in eachindex(K)
                @inbounds @views K[i] = _kernel(κ, X[i,:],X[i,:])
            end
        else
            for i in eachindex(K)
                @inbounds @views K[i] = _kernel(κ,X[:,i],X[:,i])
            end
        end
        return K
end

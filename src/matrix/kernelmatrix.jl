"""
```
    kernelmatrix!(K::Matrix, κ::Kernel, X::Matrix; obsdim::Integer=2, symmetrize::Bool=true)
```
In-place version of `kernelmatrix` where pre-allocated matrix `K` will be overwritten with the kernel matrix.
"""
function kernelmatrix!(
        K::Matrix,
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
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
        K::AbstractMatrix,
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix;
        obsdim::Int = defaultobs
        )
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
function kernel(κ::Kernel, x::Real, y::Real)
    kernel(κ, [x], [y])
end

function kernel(
        κ::Kernel,
        x::AbstractVector,
        y::AbstractVector;
        obsdim::Int = defaultobs
    )
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
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs,
        symmetrize::Bool = true
    )
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
        κ::Kernel,
        X::AbstractMatrix,
        Y::AbstractMatrix;
        obsdim=defaultobs
    )
    if !check_dims(X,Y,feature_dim(obsdim),obsdim)
        throw(DimensionMismatch("X $(size(X)) and Y $(size(Y)) do not have the same number of features on the dimension : $(feature_dim(obsdim))"))
    end
    _kernelmatrix(κ,X,Y,obsdim)
end

@inline _kernelmatrix(κ,X,Y,obsdim) = map(x->kappa(κ,x),pairwise(metric(κ),transform(κ,X,obsdim),transform(κ,Y,obsdim),dims=obsdim))

"""
```
    kerneldiagmatrix(κ::Kernel, X::Matrix; obsdim::Int=2)
```
Calculate the diagonal matrix of `X` with respect to kernel `κ`
`obsdim=1` means the matrix `X` has size #samples x #dimension
`obsdim=2` means the matrix `X` has size #dimension x #samples
"""
function kerneldiagmatrix(
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
        if obsdim == 1
            [@views kernel(κ,X[i,:],X[i,:]) for i in 1:size(X,obsdim)]
        elseif obsdim == 2
            [@views kernel(κ,X[:,i],X[:,i]) for i in 1:size(X,obsdim)]
        end
end

"""
```
    kerneldiagmatrix!(K::AbstractVector,κ::Kernel, X::Matrix; obsdim::Int=2)
```
In place version of `kerneldiagmatrix`
"""
function kerneldiagmatrix!(
        K::AbstractVector,
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
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

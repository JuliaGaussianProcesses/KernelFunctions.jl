# Macro for checking arguments
macro check_args(K, param, cond, desc=string(cond))
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(K)), ": ", $(string(param)), " = ", $(esc(param)), " does not ",
                "satisfy the constraint ", $(string(desc)), ".")))
        end
    end
end

function vec_of_vecs(X::AbstractMatrix; obsdim::Int = 2)
    @assert obsdim ∈ (1, 2) "obsdim should be 1 or 2, see docs of kernelmatrix"
    if obsdim == 1
        RowVecs(X)
    else
        ColVecs(X)
    end
end

dim(x::AbstractVector{<:Real}) = 1

"""
    ColVecs(X::AbstractMatrix)

A lightweight wrapper for an `AbstractMatrix` to make it behave like a vector of vectors.
Each vector represents a column of the matrix
"""
struct ColVecs{T, TX<:AbstractMatrix{T}, S} <: AbstractVector{S}
    X::TX
    function ColVecs(X::TX) where {T, TX<:AbstractMatrix{T}}
        S = typeof(view(X, :, 1))
        new{T, TX, S}(X)
    end
end

Base.size(D::ColVecs) = (size(D.X, 2),)
Base.getindex(D::ColVecs, i::Int) = view(D.X, :, i)
Base.getindex(D::ColVecs, i::CartesianIndex{1}) = view(D.X, :, i)
Base.getindex(D::ColVecs, i) = ColVecs(view(D.X, :, i))

dim(x::ColVecs) = size(x.X, 1)

Distances.pairwise(d::PreMetric, x::ColVecs) = pairwise(d, x.X; dims=2)
Distances.pairwise(d::PreMetric, x::ColVecs, y::ColVecs) = pairwise(d, x.X, y.X; dims=2)
function Distances.pairwise!(out::AbstractMatrix, d::PreMetric, x::ColVecs)
    return pairwise!(out, d, x.X; dims=2)
end
function Distances.pairwise!(out::AbstractMatrix, d::PreMetric, x::ColVecs, y::ColVecs)
    return pairwise!(out, d, x.X, y.X; dims=2)
end

"""
    RowVecs(X::AbstractMatrix)

A lightweight wrapper for an `AbstractMatrix` to make it behave like a vector of vectors.
Each vector represents a row of the matrix
"""
struct RowVecs{T, TX<:AbstractMatrix{T}, S} <: AbstractVector{S}
    X::TX
    function RowVecs(X::TX) where {T, TX<:AbstractMatrix{T}}
        S = typeof(view(X, 1, :))
        new{T, TX, S}(X)
    end
end

Base.size(D::RowVecs) = (size(D.X, 1),)
Base.getindex(D::RowVecs, i::Int) = view(D.X, i, :)
Base.getindex(D::RowVecs, i::CartesianIndex{1}) = view(D.X, i, :)
Base.getindex(D::RowVecs, i) = RowVecs(view(D.X, i, :))

dim(x::RowVecs) = size(x.X, 2)

Distances.pairwise(d::PreMetric, x::RowVecs) = pairwise(d, x.X; dims=1)
Distances.pairwise(d::PreMetric, x::RowVecs, y::RowVecs) = pairwise(d, x.X, y.X; dims=1)
function Distances.pairwise!(out::AbstractMatrix, d::PreMetric, x::RowVecs)
    return pairwise!(out, d, x.X; dims=1)
end
function Distances.pairwise!(out::AbstractMatrix, d::PreMetric, x::RowVecs, y::RowVecs)
    return pairwise!(out, d, x.X, y.X; dims=1)
end

"""
Will be implemented at some point
```julia
    params(k::Kernel)
    params(t::Transform)
```
For a kernel return a tuple with parameters of the transform followed by the specific parameters of the kernel
For a transform return its parameters, for a `ChainTransform` return a vector of `params(t)`.
"""
#params


function validate_inplace_dims(K::AbstractMatrix, x::AbstractVector, y::AbstractVector)
    validate_dims(x, y)
    if size(K) != (length(x), length(y))
        throw(DimensionMismatch(
            "Size of the target matrix K ($(size(K))) not consistent with lengths of " *
            "inputs x ($(length(x))) and y ($(length(y)))",
        ))
    end
end

function validate_inplace_dims(K::AbstractMatrix, x::AbstractVector)
    return validate_inplace_dims(K, x, x)
end

function validate_inplace_dims(K::AbstractVector, x::AbstractVector)
    if length(K) != length(x)
        throw(DimensionMismatch(
            "Length of target vector K ($(length(K))) not consistent with length of input" *
            "vector x ($(length(x))",
        ))
    end
end

function validate_dims(x::AbstractVector, y::AbstractVector)
    if dim(x) != dim(y)
        throw(DimensionMismatch(
            "Dimensionality of x ($(dim(x))) not equality to that of y ($(dim(y)))",
        ))
    end
end


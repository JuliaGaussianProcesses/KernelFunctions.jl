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


# abstract type VecOfVecs{T, TX <: AbstractMatrix{T}, S} <: AbstractVector{S} end

function vec_of_vecs(X::AbstractMatrix, obsdim)
    @assert obsdim ∈ (1, 2) "obsdim should be 1 or 2"
    if obsdim == 1
        RowVecs(X)
    else
        ColVecs(X)
    end
end

"""
    ColVecs(X::AbstractMatrix)

A lightweight wrapper for an `AbstractMatrix` to make it behave like a vector of vectors.
Each vector represents a colum of the matrix
"""
struct ColVecs{T, TX<:AbstractMatrix{T}, S} <: AbstractVector{S} # VecOfVecs{T, TX, S}
    X::TX
    function ColVecs(X::TX) where {T, TX<:AbstractMatrix{T}}
        S = typeof(view(X, :, 1))
        new{T, TX, S}(X)
    end
end

Base.size(D::ColVecs) = (size(D.X, 2),)
Base.getindex(D::ColVecs, i::Int) = view(D.X, :, i)
Base.getindex(D::ColVecs, i) = ColVecs(view(D.X, :, i))

"""
    RowVecs(X::AbstractMatrix)

A lightweight wrapper for an `AbstractMatrix` to make it behave like a vector of vectors.
Each vector represents a colum of the matrix
"""
struct RowVecs{T, TX<:AbstractMatrix{T}, S} <: AbstractVector{S} # VecOfVecs{T, TX, S}
    X::TX
    function RowVecs(X::TX) where {T, TX<:AbstractMatrix{T}}
        S = typeof(view(X, 1, :))
        new{T, TX, S}(X)
    end
end

Base.size(D::RowVecs) = (size(D.X, 1),)
Base.getindex(D::RowVecs, i::Int) = view(D.X, i, :)
Base.getindex(D::RowVecs, i) = RowVecs(view(D.X, i, :))

# Take highest Float among possibilities
# function promote_float(Tₖ::DataType...)
#     if length(Tₖ) == 0
#         return Float64
#     end
#     T = promote_type(Tₖ...)
#     return T <: Real ? T : Float64
# end

check_dims(K,X,Y,featdim,obsdim) = check_dims(X,Y,featdim,obsdim) && (size(K) == (size(X,obsdim),size(Y,obsdim)))

check_dims(X,Y,featdim,obsdim) = size(X,featdim) == size(Y,featdim)


feature_dim(obsdim::Int) = obsdim == 1 ? 2 : 1

base_kernel(k::Kernel) = eval(nameof(typeof(k)))

base_transform(t::Transform) = eval(nameof(typeof(t)))

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

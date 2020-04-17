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


"""
    ColVecs{T, TX<:AbstractMatrix}

A lightweight box for an `AbstractMatrix` to make it behave like a vector of vectors.
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
Base.getindex(D::ColVecs, i::AbstractVector{Int}) = ColVecs(view(D.X, :, i))

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

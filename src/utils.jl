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
struct ColVecs{T, TX<:AbstractMatrix{T}} <: AbstractVector{Vector{T}}
    X::TX
    ColVecs(X::TX) where {T, TX<:AbstractMatrix{T}} = new{T, TX}(X)
end

Base.:(==)(D1::ColVecs, D2::ColVecs) = D1.X == D2.X
Base.size(D::ColVecs) = (size(D.X, 2),)
Base.length(D::ColVecs) = size(D.X, 2)
Base.getindex(D::ColVecs, n::Int) = D.X[:, n]
Base.getindex(D::ColVecs, n::CartesianIndex{1}) = getindex(D, n[1])
Base.getindex(D::ColVecs, n) = ColVecs(D.X[:, n])
Base.view(D::ColVecs, n::Int) = view(D.X, :, n)
Base.view(D::ColVecs, n) = ColVecs(view(D.X, :, n))
Base.eltype(D::ColVecs{T}) where T = Vector{T}
Base.zero(D::ColVecs) = ColVecs(zero(D.X))
Base.iterate(D::ColVecs) = (view(D.X, :, 1), 2)
Base.iterate(D::ColVecs, state) = state > length(D) ? nothing : (view(D.X, :, state), state + 1)


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

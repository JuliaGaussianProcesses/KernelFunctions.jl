"""
    lazykernelmatrix(κ::Kernel, x::AbstractVector) -> AbstractMatrix

Construct a lazy representation of the kernel `κ` for each pair of inputs in `x`.

The result is a matrix with the same entries as [`kernelmatrix(κ, x)`](@ref) but where the
entries are not computed until they are needed.
"""
lazykernelmatrix(κ::Kernel, x) = lazykernelmatrix(κ, x, x)

"""
    lazykernelmatrix(κ::Kernel, x::AbstractVector, y::AbstractVector) -> AbstractMatrix

Construct a lazy representation of the kernel `κ` for each pair of inputs in `x`.

The result is a matrix with the same entries as [`kernelmatrix(κ, x, y)`](@ref) but where
the entries are not computed until they are needed.
"""
lazykernelmatrix(κ::Kernel, x, y) = LazyKernelMatrix(κ, x, y)

"""
    LazyKernelMatrix(κ::Kernel, x[, y])
    LazyKernelMatrix{T<:Real}(κ::Kernel, x, y)

Construct a lazy representation of the kernel `κ` for each pair of inputs in `x` and `y`.

Instead of constructing this directly, it is better to call
[`lazykernelmatrix(κ, x[, y])`](@ref lazykernelmatrix).
"""
struct LazyKernelMatrix{T<:Real,Tk<:Kernel,Tx<:AbstractVector,Ty<:AbstractVector} <:
       AbstractMatrix{T}
    kernel::Tk
    x::Tx
    y::Ty
    function LazyKernelMatrix{T}(κ::Tk, x::Tx, y::Ty) where {T<:Real,Tk<:Kernel,Tx,Ty}
        Base.require_one_based_indexing(x)
        Base.require_one_based_indexing(y)
        return new{T,Tk,Tx,Ty}(κ, x, y)
    end
    function LazyKernelMatrix{T}(κ::Tk, x::Tx) where {T<:Real,Tk<:Kernel,Tx}
        Base.require_one_based_indexing(x)
        return new{T,Tk,Tx,Tx}(κ, x, x)
    end
end
function LazyKernelMatrix(κ::Kernel, x::AbstractVector, y::AbstractVector)
    # evaluate once to get eltype
    T = typeof(κ(first(x), first(y)))
    return LazyKernelMatrix{T}(κ, x, y)
end
LazyKernelMatrix(κ::Kernel, x::AbstractVector) = LazyKernelMatrix(κ, x, x)

Base.Matrix(K::LazyKernelMatrix) = kernelmatrix(K.kernel, K.x, K.y)
function Base.AbstractMatrix{T}(K::LazyKernelMatrix) where {T}
    return LazyKernelMatrix{T}(K.kernel, K.x, K.y)
end

Base.size(K::LazyKernelMatrix) = (length(K.x), length(K.y))

Base.axes(K::LazyKernelMatrix) = (axes(K.x, 1), axes(K.y, 1))

function Base.getindex(K::LazyKernelMatrix{T}, i::Int, j::Int) where {T}
    return T(K.kernel(K.x[i], K.y[j]))
end
for f in (:getindex, :view)
    @eval begin
        function Base.$f(
            K::LazyKernelMatrix{T},
            I::Union{Colon,AbstractVector},
            J::Union{Colon,AbstractVector},
        ) where {T}
            return LazyKernelMatrix{T}(K.kernel, $f(K.x, I), $f(K.y, J))
        end
    end
end

Base.zero(K::LazyKernelMatrix{T}) where {T} = LazyKernelMatrix{T}(ZeroKernel(), K.x, K.y)
Base.one(K::LazyKernelMatrix{T}) where {T} = LazyKernelMatrix{T}(WhiteKernel(), K.x, K.y)

function Base.:*(c::S, K::LazyKernelMatrix{T}) where {T,S<:Real}
    R = typeof(oneunit(S) * oneunit(T))
    return LazyKernelMatrix{R}(c * K.kernel, K.x, K.y)
end
Base.:*(K::LazyKernelMatrix, c::Real) = c * K
Base.:/(K::LazyKernelMatrix, c::Real) = K * inv(c)
Base.:\(c::Real, K::LazyKernelMatrix) = inv(c) * K

function Base.:+(K::LazyKernelMatrix{T}, C::UniformScaling{S}) where {T,S<:Real}
    if isequal(K.x, K.y)
        R = typeof(zero(T) + zero(S))
        return LazyKernelMatrix{R}(K.kernel + C.λ * WhiteKernel(), K.x, K.y)
    else
        return Matrix(K) + C
    end
end
function Base.:+(C::UniformScaling{S}, K::LazyKernelMatrix{T}) where {T,S<:Real}
    if isequal(K.x, K.y)
        R = typeof(zero(T) + zero(S))
        return LazyKernelMatrix{R}(C.λ * WhiteKernel() + K.kernel, K.x, K.y)
    else
        return C + Matrix(K)
    end
end
function Base.:+(K1::LazyKernelMatrix, K2::LazyKernelMatrix)
    if isequal(K1.x, K2.x) && isequal(K1.y, K2.y)
        return LazyKernelMatrix(K1.kernel + K2.kernel, K1.x, K1.y)
    else
        return Matrix(K1) + Matrix(K2)
    end
end

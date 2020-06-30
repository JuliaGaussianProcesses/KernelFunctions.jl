struct ANOVAKernel{T<:Real} <: BaseKernel
    d::Vector{T}
    function ANOVAKernel(; d::T=1.0) where {T<:Real}
        @assert d > 0 "ANOVAKernel: Given degree d is invalid."
        return new{T}([d])
    end
end

function (κ::ANOVAKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    return sum(exp.( - (x .- y).^2 ) .^ first(κ.d))
end

(κ::ANOVAKernel)(x::Real, y::Real) = exp( - (x - y)^2 ) ^ first(κ.d)

Base.show(io::IO, κ::ANOVAKernel) = print(io, "ANOVA Kernel (d = ", first(κ.d), ")")

_dist(κ::ANOVAKernel, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = exp.( - (x .- x').^2 ) .^ first(κ.d)

function _dist(κ::ANOVAKernel, x::ColVecs, y::ColVecs)
    k = zeros(eltype(x), first(size(x)), first(size(y)))
    for d ∈ dim(x)
        colx = view(x.X, d, :)
        coly = view(y.X, d, :)
        k += exp.( - (colx .- coly').^2 ) .^ first(κ.d)
    end
    return k
end

function _dist(κ::ANOVAKernel, x::RowVecs, y::RowVecs)
    k = zeros(eltype(x), first(size(x)), first(size(y)))
    for d ∈ dim(x)
        colx = view(x.X, :, d)
        coly = view(y.X, :, d)
        k += exp.( - (colx .- coly').^2 ) .^ first(κ.d)
    end
    return k
end

function kernelmatrix(κ::ANOVAKernel, x::AbstractVector)
    return _dist(κ, x, x)
end

function kernelmatrix!(K::AbstractMatrix, κ::ANOVAKernel, x::AbstractVector)
    K .= _dist(κ, x, x)
    return K
end

function kernelmatrix(κ::ANOVAKernel, x::AbstractVector, y::AbstractVector)
    return _dist(κ, x, y)
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::ANOVAKernel,
    x::AbstractVector,
    y::AbstractVector,
)
    K .= _dist(κ, x, y)
    return K
end

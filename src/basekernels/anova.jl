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

(κ::ANOVAKernel)(x::Real, y::Real) = exp( - (x - y)^2 )^first(κ.d)

Base.show(io::IO, κ::ANOVAKernel) = print(io, "ANOVA Kernel d = ", first(κ.d), ")")

function kernelmatrix(κ::ANOVAKernel, x::AbstractVector)
    k = zeros(eltype(x), dim(x), dim(x))
    for d ∈ size(x)
        col = reshape(x[d], dim(x), 1)
        k += exp( - (col .- col').^2 ) .^ first(κ.d)
    end
    return k
end

function kernelmatrix!(K::AbstractMatrix, κ::ANOVAKernel, x::AbstractVector)
    for d ∈ size(x)
        col = reshape(x[d], dim(x), 1)
        K += exp( - (col .- col').^2 ) .^ first(κ.d)
    end
    return K
end

function kernelmatrix(κ::ANOVAKernel, x::AbstractVector, y::AbstractVector)
    k = zeros(eltype(x), dim(x), dim(y))
    for d ∈ size(x)
        colx = reshape(x[d], dim(x), 1)
        coly = reshape(y[d], dim(y), 1)
        k += exp( - (colx .- coly').^2 ) .^ first(κ.d)
    end
    return k
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::ANOVAKernel,
    x::AbstractVector,
    y::AbstractVector,
)
    for d ∈ size(x)
        colx = reshape(x[d], dim(x), 1)
        coly = reshape(y[d], dim(y), 1)
        K += exp( - (colx .- coly').^2 ) .^ first(κ.d)
    end
    return K
end

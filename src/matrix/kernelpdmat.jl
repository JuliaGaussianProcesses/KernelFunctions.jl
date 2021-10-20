using .PDMats: PDMat

"""
    kernelmatrix(::Type{<:PDMats.PDMat}, k::Kernel, x::AbstractVector)

Compute the positive-definite `kernelmatrix` as a `PDMats.PDMat` matrix with the
Cholesky decomposition precomputed.

The algorithm adds a diagonal "nugget" term to the kernel matrix which is increased until
positive definiteness is achieved. The algorithm gives up with an error if the nugget
becomes larger than 1% of the largest value in the kernel matrix.

See also: [PDMats.jl](https://github.com/JuliaStats/PDMats.jl)
"""
function kernelmatrix(::Type{T}, κ::Kernel, X::AbstractVector) where {T<:PDMat}
    K = kernelmatrix(κ, X)
    threshold = maximum(K) / 100
    α = eps(eltype(K))
    while !isposdef(K + α * I) && α < threshold
        α *= 2
    end
    if α >= threshold
        error(
            "Adding noise on the diagonal was not sufficient to build a positive-definite" *
            " matrix:\n\t- Check that your kernel parameters are not extreme\n\t- Check" *
            " that your data is sufficiently sparse\n\t- Maybe use a different kernel",
        )
    end
    return T(K + α * I)
end

# deprecations
Base.@deprecate kernelpdmat(κ::Kernel, X::AbstractVector) kernelmatrix(PDMat, κ, X)
Base.@deprecate kernelpdmat(κ::Kernel, X::AbstractMatrix; obsdim=defaultobs) kernelmatrix(
    PDMat, κ, X; obsdim=obsdim
)

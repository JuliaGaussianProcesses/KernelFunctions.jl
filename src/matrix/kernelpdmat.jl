using .PDMats: PDMat

export kernelpdmat

"""
    kernelpdmat(k::Kernel, X::AbstractMatrix; obsdim)
    kernelpdmat(k::Kernel, X::AbstractVector)

Compute a positive-definite matrix in the form of a `PDMat` matrix (see [PDMats.jl](https://github.com/JuliaStats/PDMats.jl)),
with the Cholesky decomposition precomputed.
The algorithm adds a diagonal "nugget" term to the kernel matrix which is increased until positive
definiteness is achieved. The algorithm gives up with an error if the nugget becomes larger than 1% of the largest value in the kernel matrix.
"""
function kernelpdmat(κ::Kernel, X::AbstractMatrix; obsdim::Union{Int,Nothing}=defaultobs)
    return kernelpdmat(κ, vec_of_vecs(X; obsdim=obsdim))
end

function kernelpdmat(κ::Kernel, X::AbstractVector)
    K = kernelmatrix(κ, X)
    Kmax = maximum(K)
    α = eps(eltype(K))
    while !isposdef(K + α * I) && α < 0.01 * Kmax
        α *= 2.0
    end
    if α >= 0.01 * Kmax
        error(
            "Adding noise on the diagonal was not sufficient to build a positive-definite" *
            " matrix:\n\t- Check that your kernel parameters are not extreme\n\t- Check" *
            " that your data is sufficiently sparse\n\t- Maybe use a different kernel",
        )
    end
    return PDMat(K + α * I)
end

using .PDMats: PDMat

export kernelpdmat

"""
    kernelpdmat(k::Kernel, X::AbstractMatrix; obsdim::Int=2)
    kernelpdmat(k::Kernel, X::AbstractVector)

Compute a positive-definite matrix in the form of a `PDMat` matrix see [PDMats.jl](https://github.com/JuliaStats/PDMats.jl)
with the cholesky decomposition precomputed.
The algorithm recursively tries to add recursively a diagonal nugget until positive
definiteness is achieved or until the noise is too big.
"""
function kernelpdmat(κ::Kernel, X::AbstractMatrix; obsdim::Int=defaultobs)
    kernelpdmat(κ, vec_of_vecs(X; obsdim=obsdim))
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

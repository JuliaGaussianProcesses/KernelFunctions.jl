"""
    Guarantees to return a positive-definite matrix in the form of a `PDMat` matrix with the cholesky decomposition precomputed
"""
function kernelpdmat(
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
    K = kernelmatrix(κ,X,obsdim=obsdim)
    α = eps(eltype(K))
    while !isposdef(K+αI) && α < 0.01*maximum(K)
        α *= 2.0
    end
    if α >= 0.01*maximum(K)
        @error "Adding noise on the diagonal was not sufficient to build a positive-definite matrix:\n - Check that your kernel parameters are not extreme\n - Check that your data is sufficiently sparse\n - Maybe use a different kernel"
    end
    return PDMat(K+αI)
end

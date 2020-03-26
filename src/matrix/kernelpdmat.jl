using .PDMats: PDMat

export kernelpdmat

"""
    Compute a positive-definite matrix in the form of a `PDMat` matrix see [PDMats.jl]() with the cholesky decomposition precomputed
    The algorithm recursively tries to add recursively a diagonal nugget until positive definiteness is achieved or that the noise is too big
"""
function kernelpdmat(
        κ::Kernel,
        X::AbstractMatrix;
        obsdim::Int = defaultobs
        )
    K = kernelmatrix(κ,X,obsdim=obsdim)
    Kmax =maximum(K)
    α = eps(eltype(K))
    while !isposdef(K+α*I) && α < 0.01*Kmax
        α *= 2.0
    end
    if α >= 0.01*Kmax
        throw(ErrorException("Adding noise on the diagonal was not sufficient to build a positive-definite matrix:\n\t- Check that your kernel parameters are not extreme\n\t- Check that your data is sufficiently sparse\n\t- Maybe use a different kernel"))
    end
    return PDMat(K+α*I)
end

kernelpdmat(κ::Kernel,X::AbstractVector{<:Real};obsdim=defaultobs) = kernelpdmat(κ,reshape(X,1,:),obsdim=2)

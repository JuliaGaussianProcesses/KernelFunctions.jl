function kernelkronmat(
    κ::Kernel,
    X::AbstractVector,
    dims::Int
    )
    @assert iskroncompatible(κ) "The kernel chosed is not compatible for kroenecker matrices"
    K = kernelmatrix(κ,reshape(X,:,1),obsdim=1)

end


function iskroncompatible(κ::Kernel)

end

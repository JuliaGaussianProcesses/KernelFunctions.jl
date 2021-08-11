"""
    MOKernel

Abstract type for kernels with multiple outpus.
"""
abstract type MOKernel <: Kernel end

"""
    matrixkernel

Convenience function ... More documentation soon. 
"""
function matrixkernel(k::MOK, x, y; outputsize) where {T,MOK<:MOKernel}
    @assert size(x) == size(y)
    xMO = MOInputIsotopicByFeatures([x], outputsize)
    yMO = MOInputIsotopicByFeatures([y], outputsize)
    return kernelmatrix(k, xMO, yMO)
end

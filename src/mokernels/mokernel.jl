"""
    MOKernel

Abstract type for kernels with multiple outpus.
"""
abstract type MOKernel <: Kernel end

"""
    matrixkernel(k::MOK, x, y)
    matrixkernel(k::IndependentMOKernel, x, y(; outputsize))

Convenience function to compute the matrix kernel for two inputs `x` and `y`. The `outputsize` keyword is only required for the `IndependentMOKernel` to indicated the number of outputs. 
"""
function matrixkernel(k::MOKernel, x, y; outputsize)
    @assert size(x) == size(y)
    xMO = MOInputIsotopicByFeatures([x], outputsize)
    yMO = MOInputIsotopicByFeatures([y], outputsize)
    return kernelmatrix(k, xMO, yMO)
end

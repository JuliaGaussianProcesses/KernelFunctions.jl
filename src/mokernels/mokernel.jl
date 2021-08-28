"""
    MOKernel

Abstract type for kernels with multiple outpus.
"""
abstract type MOKernel <: Kernel end

"""
    matrixkernel(k::MOK, x, y)

Convenience function to compute the matrix kernel for two inputs `x` and `y`. The `outputsize` keyword is only required for the `IndependentMOKernel` to indicated the number of outputs. 
"""
function matrixkernel(k::MOKernel, x, y, out_dim)
    @assert size(x) == size(y)
    xMO = MOInputIsotopicByFeatures([x], out_dim)
    yMO = MOInputIsotopicByFeatures([y], out_dim)
    return kernelmatrix(k, xMO, yMO)
end

function matrixkernel(k::MOKernel, x, y)
    return throw(
        ArgumentError(
            "This kernel does not have a specific matrixkernel implementation, you can call `matrixkernel(k, x, y, out_dim)`",
        ),
    )
end

function _kernelmatrix_kron_helper(::MOInputIsotopicByFeatures, Kfeatures, Koutputs)
    return kron(Kfeatures, Koutputs)
end

function _kernelmatrix_kron_helper(::MOInputIsotopicByOutputs, Kfeatures, Koutputs)
    return kron(Koutputs, Kfeatures)
end

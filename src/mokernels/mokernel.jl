"""
    MOKernel

Abstract type for kernels with multiple outpus.
"""
abstract type MOKernel <: Kernel end

function _kernelmatrix_kron_helper(::MOInputIsotopicByFeatures, Kfeatures, Koutputs)
    return kron(Kfeatures, Koutputs)
end

function _kernelmatrix_kron_helper(::MOInputIsotopicByOutputs, Kfeatures, Koutputs)
    return kron(Koutputs, Kfeatures)
end

if VERSION >= v"1.6"
    function _kernelmatrix_kron_helper!(K, ::MOInputIsotopicByFeatures, Kfeatures, Koutputs)
        return kron!(K, Kfeatures, Koutputs)
    end

    function _kernelmatrix_kron_helper!(K, ::MOInputIsotopicByOutputs, Kfeatures, Koutputs)
        return kron!(K, Koutputs, Kfeatures)
    end
end

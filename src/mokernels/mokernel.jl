"""
    MOKernel

Abstract type for kernels with multiple outpus.
"""
abstract type MOKernel <: Kernel end

function _kernelmatrix_kron_helper(::Type{<:MOInputIsotopicByFeatures}, Kfeatures, Koutputs)
    return kron(Kfeatures, Koutputs)
end

function _kernelmatrix_kron_helper(::Type{<:MOInputIsotopicByOutputs}, Kfeatures, Koutputs)
    return kron(Koutputs, Kfeatures)
end

if VERSION >= v"1.6"
    function _kernelmatrix_kron_helper!(
        K, ::Type{<:MOInputIsotopicByFeatures}, Kfeatures, Koutputs
    )
        return kron!(K, Kfeatures, Koutputs)
    end

    function _kernelmatrix_kron_helper!(
        K, ::Type{<:MOInputIsotopicByOutputs}, Kfeatures, Koutputs
    )
        return kron!(K, Koutputs, Kfeatures)
    end
end

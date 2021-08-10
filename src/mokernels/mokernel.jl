"""
    MOKernel

Abstract type for kernels with multiple outpus.
"""
abstract type MOKernel <: Kernel end

abstract type KroneckerKernelMatrix end
struct LazyKroneckerKernelMatrix <: KroneckerKernelMatrix end
struct ExplicitKroneckerKernelMatrix <: KroneckerKernelMatrix end

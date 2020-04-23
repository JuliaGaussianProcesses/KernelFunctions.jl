
"""
Abstract type defining a slice-wise transformation on an input matrix
"""
abstract type Transform end

abstract type Kernel end
abstract type BaseKernel <: Kernel end
abstract type SimpleKernel <: BaseKernel end

(k::SimpleKernel)(x, y) = kappa(k, evaluate(metric(k), x, y))

module KernelFunctions

export kernelmatrix, kernelmatrix!, kerneldiagmatrix, kerneldiagmatrix!, kappa, kernelpdmat
export get_params, set_params!

export Kernel
export ConstantKernel, WhiteKernel, ZeroKernel
export SqExponentialKernel, ExponentialKernel, GammaExponentialKernel
export ExponentiatedKernel
export MaternKernel, Matern32Kernel, Matern52Kernel
export LinearKernel, PolynomialKernel
export RationalQuadraticKernel, GammaRationalQuadraticKernel
export KernelSum, KernelProduct

export SelectTransform, ChainTransform, ScaleTransform, LowRankTransform, IdentityTransform, FunctionTransform


using Distances, LinearAlgebra
using SpecialFunctions: lgamma, besselk
using StatsFuns: logtwo
using PDMats: PDMat

const defaultobs = 2

"""
Abstract type defining a slice-wise transformation on an input matrix
"""
abstract type Transform end
abstract type Kernel{T,Tr<:Transform} end

include("utils.jl")
include("distances/dotproduct.jl")
include("distances/delta.jl")
include("transform/transform.jl")
kernels = ["exponential","matern","polynomial","constant","rationalquad","exponentiated"]
for k in kernels
    include(joinpath("kernels",k*".jl"))
end
include("matrix/kernelmatrix.jl")
include("matrix/kernelpdmat.jl")
include("kernels/kernelsum.jl")
include("kernels/kernelproduct.jl")

include("generic.jl")
include("squeeze.jl")

end

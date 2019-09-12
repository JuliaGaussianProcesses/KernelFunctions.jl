module KernelFunctions

export kernelmatrix, kernelmatrix!, kerneldiagmatrix, kerneldiagmatrix!, kappa
export Kernel
export SqExponentialKernel, ExponentialKernel, GammaExponentialKernel
export MaternKernel, Matern32Kernel, Matern52Kernel
export LinearKernel, PolynomialKernel
export ConstantKernel, WhiteKernel, ZeroKernel

export Transform, ScaleTransform

using Distances, LinearAlgebra
using Zygote: @adjoint
using SpecialFunctions: lgamma, besselk
using StatsFuns: logtwo

const defaultobs = 2
abstract type Kernel{T,Tr<:Transform} end

include("zygote_rules.jl")
include("utils.jl")
include("distances/dotproduct.jl")
include("distances/delta.jl")
include("transform/transform.jl")
include("kernelmatrix.jl")

kernels = ["exponential","matern","polynomial","constant","rationalquad"]
for k in kernels
    include(joinpath("kernels",k*".jl"))
end

include("generic.jl")


end

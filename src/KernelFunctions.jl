module KernelFunctions

export kernelmatrix, kernelmatrix!, kerneldiagmatrix, kerneldiagmatrix!, kappa
export Kernel
export SqExponentialKernel, ExponentialKernel, GammaExponentialKernel
export MaternKernel, Matern32Kernel, Matern52Kernel
export LinearKernel, PolynomialKernel
export ConstantKernel, WhiteKernel, ZeroKernel



using Distances, LinearAlgebra
using Zygote: @adjoint
using SpecialFunctions: lgamma, besselk
using StatsFuns: logtwo

const defaultobs = 2

# include("zygote_rules.jl")
include("utils.jl")
include("distances/dotproduct.jl")
include("distances/delta.jl")
include("transform/transform.jl")


abstract type Kernel{T,Tr<:Transform} end

kernels = ["exponential","matern","polynomial","constant","rationalquad","exponentiated"]
for k in kernels
    include(joinpath("kernels",k*".jl"))
end
include("kernelmatrix.jl")

include("generic.jl")


end

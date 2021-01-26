module KernelFunctions

if !isfile(joinpath(@__DIR__, "update_v0.8.0"))
    printstyled(
        stdout,
        """
        WARNING: SqExponentialKernel changed convention in version 0.8.0.
        This kernel now divides the squared distance by 2 to align with standard practice.
        This warning will be removed in 0.9.0.
        """;
        color=Base.info_color(),
    )
    touch(joinpath(@__DIR__, "update_v0.8.0"))
end

export kernelmatrix, kernelmatrix!, kerneldiagmatrix, kerneldiagmatrix!
export transform
export duplicate, set! # Helpers

export Kernel, MOKernel
export ConstantKernel, WhiteKernel, EyeKernel, ZeroKernel, WienerKernel
export CosineKernel
export SqExponentialKernel, RBFKernel, GaussianKernel, SEKernel
export LaplacianKernel, ExponentialKernel, GammaExponentialKernel
export ExponentiatedKernel
export FBMKernel
export MaternKernel, Matern12Kernel, Matern32Kernel, Matern52Kernel
export LinearKernel, PolynomialKernel
export RationalQuadraticKernel, GammaRationalQuadraticKernel
export GaborKernel, PiecewisePolynomialKernel
export PeriodicKernel, NeuralNetworkKernel
export KernelSum, KernelProduct, KernelTensorProduct
export TransformedKernel, ScaledKernel

export Transform,
    SelectTransform,
    ChainTransform,
    ScaleTransform,
    LinearTransform,
    ARDTransform,
    IdentityTransform,
    FunctionTransform,
    PeriodicTransform

export NystromFact, nystrom

export spectral_mixture_kernel, spectral_mixture_product_kernel

export ColVecs, RowVecs

export MOInput
export IndependentMOKernel, LatentFactorMOKernel

# Reexports
export tensor, ⊗

using Compat
using Requires
using Distances, LinearAlgebra
using Functors
using SpecialFunctions: loggamma, besselk, polygamma
using ZygoteRules: @adjoint, pullback
using StatsFuns: logtwo
using StatsBase
using TensorCore

abstract type Kernel end
abstract type SimpleKernel <: Kernel end

include("utils.jl")
include(joinpath("distances", "pairwise.jl"))
include(joinpath("distances", "dotproduct.jl"))
include(joinpath("distances", "delta.jl"))
include(joinpath("distances", "sinus.jl"))

include(joinpath("transform", "transform.jl"))
include(joinpath("transform", "scaletransform.jl"))
include(joinpath("transform", "ardtransform.jl"))
include(joinpath("transform", "lineartransform.jl"))
include(joinpath("transform", "functiontransform.jl"))
include(joinpath("transform", "selecttransform.jl"))
include(joinpath("transform", "chaintransform.jl"))
include(joinpath("transform", "periodic_transform.jl"))

include(joinpath("basekernels", "constant.jl"))
include(joinpath("basekernels", "cosine.jl"))
include(joinpath("basekernels", "exponential.jl"))
include(joinpath("basekernels", "exponentiated.jl"))
include(joinpath("basekernels", "fbm.jl"))
include(joinpath("basekernels", "gabor.jl"))
include(joinpath("basekernels", "matern.jl"))
include(joinpath("basekernels", "nn.jl"))
include(joinpath("basekernels", "periodic.jl"))
include(joinpath("basekernels", "piecewisepolynomial.jl"))
include(joinpath("basekernels", "polynomial.jl"))
include(joinpath("basekernels", "rationalquad.jl"))
include(joinpath("basekernels", "sm.jl"))
include(joinpath("basekernels", "wiener.jl"))

include(joinpath("kernels", "transformedkernel.jl"))
include(joinpath("kernels", "scaledkernel.jl"))
include(joinpath("matrix", "kernelmatrix.jl"))
include(joinpath("kernels", "kernelsum.jl"))
include(joinpath("kernels", "kernelproduct.jl"))
include(joinpath("kernels", "kerneltensorproduct.jl"))
include(joinpath("kernels", "overloads.jl"))
include(joinpath("approximations", "nystrom.jl"))
include("generic.jl")

include(joinpath("mokernels", "mokernel.jl"))
include(joinpath("mokernels", "moinput.jl"))
include(joinpath("mokernels", "independent.jl"))
include(joinpath("mokernels", "slfm.jl"))

include("zygote_adjoints.jl")

include("test_utils.jl")

include("deprecated.jl")

function __init__()
    @require Kronecker = "2c470bb0-bcc8-11e8-3dad-c9649493f05e" begin
        include(joinpath("matrix", "kernelkroneckermat.jl"))
    end
    @require PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150" begin
        include(joinpath("matrix", "kernelpdmat.jl"))
    end
end

end

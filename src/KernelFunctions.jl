module KernelFunctions

export kernelmatrix, kernelmatrix!, kernelmatrix_diag, kernelmatrix_diag!
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
export RationalKernel, RationalQuadraticKernel, GammaRationalKernel
export PiecewisePolynomialKernel
export PeriodicKernel, NeuralNetworkKernel
export KernelSum, KernelProduct, KernelTensorProduct
export TransformedKernel, ScaledKernel, NormalizedKernel
export GibbsKernel

export Transform,
    SelectTransform,
    ChainTransform,
    ScaleTransform,
    LinearTransform,
    ARDTransform,
    IdentityTransform,
    FunctionTransform,
    PeriodicTransform
export with_lengthscale

export median_heuristic_transform

export NystromFact, nystrom

export gaborkernel
export spectral_mixture_kernel, spectral_mixture_product_kernel

export ColVecs, RowVecs

export MOInput, prepare_isotopic_multi_output_data, prepare_heterotopic_multi_output_data
export IndependentMOKernel,
    LatentFactorMOKernel, IntrinsicCoregionMOKernel, LinearMixingModelKernel

# Reexports
export tensor, ⊗, compose

using Compat
using ChainRulesCore: ChainRulesCore, Tangent, ZeroTangent, NoTangent
using ChainRulesCore: @thunk, InplaceableThunk
using CompositionsBase
using Distances
using FillArrays
using Functors
using LinearAlgebra
using Requires
using SpecialFunctions: loggamma, besselk, polygamma
using IrrationalConstants: logtwo, twoπ, invsqrt2
using LogExpFunctions: softplus
using StatsBase
using TensorCore
using ZygoteRules: ZygoteRules, AContext, literal_getproperty, literal_getfield

# Hack to work around Zygote type inference problems.
const Distances_pairwise = Distances.pairwise

using Statistics: median!

abstract type Kernel end
abstract type SimpleKernel <: Kernel end

include("utils.jl")
include("distances/pairwise.jl")
include("distances/dotproduct.jl")
include("distances/delta.jl")
include("distances/sinus.jl")

include("transform/transform.jl")
include("transform/scaletransform.jl")
include("transform/ardtransform.jl")
include("transform/lineartransform.jl")
include("transform/functiontransform.jl")
include("transform/selecttransform.jl")
include("transform/chaintransform.jl")
include("transform/periodic_transform.jl")
include("kernels/transformedkernel.jl")
include("transform/with_lengthscale.jl")

include("basekernels/constant.jl")
include("basekernels/cosine.jl")
include("basekernels/exponential.jl")
include("basekernels/exponentiated.jl")
include("basekernels/fbm.jl")
include("basekernels/gabor.jl")
include("basekernels/matern.jl")
include("basekernels/nn.jl")
include("basekernels/periodic.jl")
include("basekernels/piecewisepolynomial.jl")
include("basekernels/polynomial.jl")
include("basekernels/rational.jl")
include("basekernels/sm.jl")
include("basekernels/wiener.jl")

include("kernels/gibbskernel.jl")
include("kernels/scaledkernel.jl")
include("kernels/normalizedkernel.jl")
include("matrix/kernelmatrix.jl")
include("kernels/kernelsum.jl")
include("kernels/kernelproduct.jl")
include("kernels/kerneltensorproduct.jl")
include("kernels/overloads.jl")
include("kernels/neuralkernelnetwork.jl")
include("approximations/nystrom.jl")
include("generic.jl")

include("mokernels/moinput.jl")
include("mokernels/mokernel.jl")
include("mokernels/independent.jl")
include("mokernels/slfm.jl")
include("mokernels/intrinsiccoregion.jl")
include("mokernels/lmm.jl")

include("chainrules.jl")

include("test_utils.jl")

function __init__()
    @require Kronecker = "2c470bb0-bcc8-11e8-3dad-c9649493f05e" begin
        include("matrix/kernelkroneckermat.jl")
    end
    @require PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150" begin
        include("matrix/kernelpdmat.jl")
    end
end

end

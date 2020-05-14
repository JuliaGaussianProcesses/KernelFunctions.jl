using KernelFunctions
using KernelFunctions: kappa, ColVecs, RowVecs
import Zygote, ForwardDiff, ReverseDiff, FiniteDifferences
using Test, LinearAlgebra, Random

include("utils_AD.jl")
ADs = [:Zygote, :ForwardDiff, :ReverseDiff]

kname = "SEKernel_lengthscale"
kfunction = () -> SEKernel()
kfunction = (l -> transform(SEKernel(), first(l)))
# args = nothing
args = [2.0]
v = test_FiniteDiff(kname, kfunction, args)
if !v.anynonpass
    for AD in ADs
        test_AD(AD, kname, kfunction, args)
    end
end

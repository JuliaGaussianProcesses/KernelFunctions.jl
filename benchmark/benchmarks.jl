using BenchmarkTools
using Random
using Distances, LinearAlgebra

const SUITE = BenchmarkGroup()

Random.seed!(1234)

dim = 50
N1 = 1000; N2 = 500;
alpha = 2.0

X = rand(Float64,N1,dim)
Y = rand(Float64,N2,dim)

KXY = rand(Float64,N1,N2)
KX = rand(Float64,N1,N1)
sKX = Symmetric(rand(Float64,N1,N1))
kX = rand(Float64,N1)

include("kernelmatrix.jl")
include("MLKernels.jl")

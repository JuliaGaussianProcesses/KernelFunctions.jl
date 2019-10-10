using Distances, LinearAlgebra
using Test
using KernelFunctions

dims = [10,5]

A = rand(dims...)
B = rand(dims...)
K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
Kdiag = [zeros(dims[1]),zeros(dims[2])]
k = SqExponentialKernel()
@testset "Inplace Kernel Matrix" begin
    for obsdim in [1,2]
        @test kernelmatrix!(K[obsdim],k,A,B,obsdim=obsdim) == kernelmatrix(k,A,B,obsdim=obsdim)
        @test kerneldiagmatrix!(Kdiag[obsdim],k,A,obsdim=obsdim) == kerneldiagmatrix(k,A,obsdim=obsdim)
    end
end

@testset "Kernel matrix" begin
    for obsdim in [1,2]
        @test kernelmatrix(k,A,B,obsdim=obsdim) == kappa.([k],pairwise(KernelFunctions.metric(k),A,B,dims=obsdim))
        @test kernelmatrix(k,A,obsdim=obsdim) == kappa.([k],pairwise(KernelFunctions.metric(k),A,dims=obsdim))
        @test k(A,B,obsdim=obsdim) == kernelmatrix(k,A,B,obsdim=obsdim)
        @test k(A,obsdim=obsdim) == kernelmatrix(k,A,obsdim=obsdim)
    end
end

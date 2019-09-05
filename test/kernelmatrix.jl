using Distances

dims = [10,5]

A = rand(dims...)
B = rand(dims...)
K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
k = SquaredExponentialKernel()
k = MaternKernel()

@testset "Inplace Kernel Matrix" begin
    for obsdim in [1,2]
        @test kernelmatrix!(K[obsdim],k,A,B,obsdim=obsdim) == kernelmatrix(k,A,B,obsdim=obsdim)
    end
end

@testset "Kernel matrix" begin
    for obsdim in [1,2]
        @test kernelmatrix(k,A,B,obsdim=obsdim) == kappa.([k],pairwise(KernelFunctions.metric(k),A,B,dims=obsdim))
    end
end

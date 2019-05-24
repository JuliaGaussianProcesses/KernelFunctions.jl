dims = [10,5]

A = rand(dims...)
B = rand(dims...)
K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
k = SquaredExponentialKernel()

@testset "Inplace kernelmatrix" begin
    for obsdim in [1,2]
        @test kernelmatrix!(K[obsdim],k,A,B,obsdim=obsdim) == kernelmatrix(k,A,B,obsdim=obsdim)
    end
end

@testset "Kernal matrix" begin
    for obsdim in [1,2]
        @test kernelmatrix(k,A,B,obsdim=obsdim) == kappa.([k],pairwise(KernelFunctions.metric(k),A,B,dims=obsdim))
    end
end

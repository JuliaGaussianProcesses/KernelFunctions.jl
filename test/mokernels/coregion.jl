@testset "coregion" begin
    rng = MersenneTwister(123)

    n_obs = 3
    in_dim = 2
    out_dim = 2
    rank = 1
    
    A = randn(out_dim, rank)
    B = A * transpose(A)

    X = [(rand(in_dim), rand(1:out_dim))  for i in 1:n_obs]

    kernel = ExponentialKernel()
    coregionkernel = CoregionMOKernel(kernel, B)
    
    @test coregionkernel isa CoregionMOKernel
    @test coregionkernel isa MOKernel
    @test coregionkernel isa Kernel
    @test coregionkernel(X[1], X[1]) isa Real
    @test coregionkernel(X[1], X[1]) ≈ B[X[1][2], X[1][2]] * kernel(X[1][1], X[1][1])

    @test kernelmatrix(coregionkernel, X, X) ≈ kernelmatrix(coregionkernel, X)

    @test string(coregionkernel) == "Coregion Multi-Output Kernel"
end

@testset "intrinsiccoregion" begin
    rng = MersenneTwister(123)

    n_obs = 3
    in_dim = 2
    out_dim = 2
    rank = 1

    A = randn(out_dim, rank)
    B = A * transpose(A) + Diagonal(randn(out_dim))

    X = [(rand(in_dim), rand(1:out_dim)) for i in 1:n_obs]

    kernel = ExponentialKernel()
    icoregionkernel = IntrinsicCoregionMOKernel(kernel, B)

    @test icoregionkernel isa IntrinsicCoregionMOKernel
    @test icoregionkernel isa MOKernel
    @test icoregionkernel isa Kernel
    @test icoregionkernel(X[1], X[1]) isa Real
    @test icoregionkernel(X[1], X[1]) ≈ B[X[1][2], X[1][2]] * kernel(X[1][1], X[1][1])

    KernelFunctions.TestUtils.test_interface(icoregionkernel, Vector{Tuple{Float64,Int}})

    @test string(coregionkernel) == "Intrinsic Coregion Multi-Output Kernel"
end

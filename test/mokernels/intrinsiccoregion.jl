@testset "intrinsiccoregion" begin
    rng = MersenneTwister(123)

    dims = (in=3, out=2, obs=3)
    rank = 1
    
    A = randn(dims.out, rank)
    B = A * transpose(A) + Diagonal(rand(dims.out))

    X = [(rand(dims.in), rand(1:dims.out)) for i in 1:dims.obs]

    kernel = ExponentialKernel()
    icoregionkernel = IntrinsicCoregionMOKernel(kernel, B)

    @test icoregionkernel isa IntrinsicCoregionMOKernel
    @test icoregionkernel isa MOKernel
    @test icoregionkernel isa Kernel
    @test icoregionkernel(X[1], X[1]) isa Real
    @test icoregionkernel(X[1], X[1]) ≈ B[X[1][2], X[1][2]] * kernel(X[1][1], X[1][1])

    KernelFunctions.TestUtils.test_interface(icoregionkernel, Vector{Tuple{Float64,Int}}, dim_out=dims.out)
    test_ADs(IntrinsicCoregionMOKernel, (kernel, B))

    @test string(coregionkernel) == "Intrinsic Coregion Multi-Output Kernel"
end

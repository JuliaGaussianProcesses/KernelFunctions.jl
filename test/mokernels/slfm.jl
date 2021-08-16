@testset "slfm" begin
    rng = MersenneTwister(123)
    FDM = FiniteDifferences.central_fdm(5, 1)
    N = 6
    in_dim = 3
    out_dim = 4
    x1IO = KernelFunctions.MOInputIsotopicByOutputs(
        [rand(rng, in_dim) for _ in 1:N], out_dim
    )
    x2IO = KernelFunctions.MOInputIsotopicByOutputs(
        [rand(rng, in_dim) for _ in 1:N], out_dim
    )
    x3IO = KernelFunctions.MOInputIsotopicByOutputs(
        [rand(rng, in_dim) for _ in 1:div(N, 2)], out_dim
    )
    x1IO = MOInput([rand(rng, in_dim) for _ in 1:N], out_dim)
    x2IO = MOInput([rand(rng, in_dim) for _ in 1:N], out_dim)

    k = LatentFactorMOKernel(
        [Matern32Kernel(), SqExponentialKernel(), FBMKernel()],
        IndependentMOKernel(GaussianKernel()),
        rand(rng, out_dim, 3),
    )
    @test k isa LatentFactorMOKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k(x1IO[1], x2IO[1]) isa Real

    @test kernelmatrix(k, x1IO, x2IO) ≈ kernelmatrix(k, collect(x1IO), collect(x2IO))

    TestUtils.test_interface(k, x1IO, x2IO, x3IO)

    x1IF = KernelFunctions.MOInputIsotopicByFeatures(x1IO.x, out_dim)
    x2IF = KernelFunctions.MOInputIsotopicByFeatures(x2IO.x, out_dim)
    x3IF = KernelFunctions.MOInputIsotopicByFeatures(x3IO.x, out_dim)

    TestUtils.test_interface(k, x1IF, x2IF, x3IF)

    @test string(k) == "Semi-parametric Latent Factor Multi-Output Kernel"
    @test repr("text/plain", k) == (
        "Semi-parametric Latent Factor Multi-Output Kernel\n" *
        "\tgᵢ: Matern 3/2 Kernel (metric = Euclidean(0.0))\n" *
        "\t\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\t\tFractional Brownian Motion Kernel (h = 0.5)\n" *
        "\teᵢ: Independent Multi-Output Kernel\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))"
    )

    # AD test
    function test_slfm(A::AbstractMatrix, x1IO, x2IO)
        k = LatentFactorMOKernel(
            [Matern32Kernel(), SqExponentialKernel(), FBMKernel()],
            IndependentMOKernel(GaussianKernel()),
            A,
        )
        return k((x1IO, 1), (x2IO, 1))
    end

    k = LatentFactorMOKernel(
        [SqExponentialKernel(), SqExponentialKernel(), SqExponentialKernel()],
        IndependentMOKernel(GaussianKernel()),
        rand(rng, out_dim, 3),
    )

    test_ADs(k)
end

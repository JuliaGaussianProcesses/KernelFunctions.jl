@testset "lmm" begin
    rng = MersenneTwister(123)
    FDM = FiniteDifferences.central_fdm(5, 1)
    N = 6
    in_dim = 3
    out_dim = 4
    x1IO = KernelFunctions.MOInputIsotopicByOutputs([rand(rng, in_dim) for _ in 1:N], out_dim)
    x2IO = KernelFunctions.MOInputIsotopicByOutputs([rand(rng, in_dim) for _ in 1:N], out_dim)
    x3IO = KernelFunctions.MOInputIsotopicByOutputs([rand(rng, in_dim) for _ in 1:div(N,2)], out_dim)
    H = rand(4, 6)
    # H = rand(4, 6) + hcat(Matrix{Float64}(I, 4,4), zeros(4,2))

    k = LinearMixingModelKernel(
        [Matern32Kernel(), SqExponentialKernel(), FBMKernel(), Matern32Kernel()], H
    )
    @test k isa LinearMixingModelKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k(x1IO[1], x2IO[1]) isa Real

    @test string(k) == "Linear Mixing Model Multi-Output Kernel"
    @test repr("text/plain", k) == (
        "Linear Mixing Model Multi-Output Kernel. Kernels:\n" *
        "\tMatern 3/2 Kernel (metric = Euclidean(0.0))\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tFractional Brownian Motion Kernel (h = 0.5)\n" *
        "\tMatern 3/2 Kernel (metric = Euclidean(0.0))"
    )

    TestUtils.test_interface(k, x1IO, x2IO, x3IO)

    x1IF = KernelFunctions.MOInputIsotopicByFeatures(x1IO.x, out_dim)
    x2IF = KernelFunctions.MOInputIsotopicByFeatures(x2IO.x, out_dim)
    x3IF = KernelFunctions.MOInputIsotopicByFeatures(x3IO.x, out_dim)

    TestUtils.test_interface(k, x1IF, x2IF, x3IF)

    a = KernelFunctions.MOInputIsotopicByOutputs([rand(rng, in_dim)], out_dim)
    b = KernelFunctions.MOInputIsotopicByOutputs([rand(rng, in_dim)], out_dim)
    @test matrixkernel(k, a.x[1], b.x[1]) ≈ k.(a, permutedims(b))

    k = LinearMixingModelKernel(SEKernel(), H)

    @test k isa LinearMixingModelKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test length(k.K) == 4
    for kernel in k.K
        @test isa(kernel, SEKernel)
    end

    @test string(k) == "Linear Mixing Model Multi-Output Kernel"
    @test repr("text/plain", k) == (
        "Linear Mixing Model Multi-Output Kernel. Kernels:\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))"
    )

    test_ADs(k)
end

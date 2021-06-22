@testset "lmm" begin
    rng = MersenneTwister(123)
    FDM = FiniteDifferences.central_fdm(5, 1)
    N = 10
    in_dim = 3
    out_dim = 6
    x1 = MOInput([rand(rng, in_dim) for _ in 1:N], out_dim)
    x2 = MOInput([rand(rng, in_dim) for _ in 1:N], out_dim)
    H = rand(4,6)

    k = NaiveLMMMOKernel(
        [Matern32Kernel(), SqExponentialKernel(), FBMKernel(), Matern32Kernel()],
        H
    )
    @test k isa NaiveLMMMOKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k(x1[1], x2[1]) isa Real

    @test string(k) == "Linear Mixing Model Multi-Output Kernel (naive implementation)"
    @test repr("text/plain", k) == (
        "Linear Mixing Model Multi-Output Kernel (naive implementation). Kernels:\n" *
        "\tMatern 3/2 Kernel (metric = Euclidean(0.0))\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tFractional Brownian Motion Kernel (h = 0.5)\n" *
        "\tMatern 3/2 Kernel (metric = Euclidean(0.0))"
    )

    k = NaiveLMMMOKernel(
        SEKernel(),
        H
    )

    @test length(k.K) == 4
    for kernel in k.K @test isa(kernel, SEKernel) end

    @test string(k) == "Linear Mixing Model Multi-Output Kernel (naive implementation)"
    @test repr("text/plain", k) == (
        "Linear Mixing Model Multi-Output Kernel (naive implementation). Kernels:\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))\n" *
        "\tSquared Exponential Kernel (metric = Euclidean(0.0))"
    )

    test_ADs(k)
end

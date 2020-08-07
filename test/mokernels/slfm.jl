@testset "slfm" begin
    x1 = MOInput([rand(5) for _ in 1:4], 2)
    x2 = MOInput([rand(5) for _ in 1:4], 2)

    k = LatentFactorMOKernel(
        [MaternKernel(), SqExponentialKernel(), FBMKernel()],
        IndependentMOKernel(GaussianKernel()),
        rand(2, 3)
    )
    @test k isa LatentFactorMOKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k(x1[2], x2[2]) isa Real

    @test kernelmatrix(k, x1, x2) == kernelmatrix(k, collect(x1), collect(x2))
    @test kernelmatrix(k, x1, x1) == kernelmatrix(k, x1)

    @test string(k) == "Semi-parametric Latent Factor Multi-Output Kernel"
    @test repr("text/plain", k) == (
        "Semi-parametric Latent Factor Multi-Output Kernel\n\tgᵢ: " *
        "Matern Kernel (ν = 1.5)\n\t\tSquared Exponential Kernel\n" * 
        "\t\tFractional Brownian Motion Kernel (h = 0.5)\n\teᵢ: " *
        "Independent Multi-Output Kernel\n\tSquared Exponential Kernel"
    )

end

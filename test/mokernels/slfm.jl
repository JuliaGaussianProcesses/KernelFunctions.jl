@testset "slfm" begin
    N = 10
    in_dim = 5
    out_dim = 4
    x1 = MOInput([rand(in_dim) for _ in 1:N], out_dim)
    x2 = MOInput([rand(in_dim) for _ in 1:N], out_dim)

    k = LatentFactorMOKernel(
        [MaternKernel(), SqExponentialKernel(), FBMKernel()],
        IndependentMOKernel(GaussianKernel()),
        rand(4, 3)
    )
    @test k isa LatentFactorMOKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k(x1[1], x2[1]) isa Real

    @test kernelmatrix(k, x1, x2) ≈ kernelmatrix(k, collect(x1), collect(x2))
    @test kernelmatrix(k, x1, x1) ≈ kernelmatrix(k, x1)

    @test string(k) == "Semi-parametric Latent Factor Multi-Output Kernel"
    @test repr("text/plain", k) == (
        "Semi-parametric Latent Factor Multi-Output Kernel\n\tgᵢ: " *
        "Matern Kernel (ν = 1.5)\n\t\tSquared Exponential Kernel\n" * 
        "\t\tFractional Brownian Motion Kernel (h = 0.5)\n\teᵢ: " *
        "Independent Multi-Output Kernel\n\tSquared Exponential Kernel"
    )

end

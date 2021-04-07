@testset "slfm" begin
    rng = MersenneTwister(123)
    FDM = FiniteDifferences.central_fdm(5, 1)
    N = 10
    in_dim = 5
    out_dim = 4
    x1 = MOInput([rand(rng, in_dim) for _ in 1:N], out_dim)
    x2 = MOInput([rand(rng, in_dim) for _ in 1:N], out_dim)

    k = LatentFactorMOKernel(
        [Matern32Kernel(), SqExponentialKernel(), FBMKernel()],
        IndependentMOKernel(GaussianKernel()),
        rand(rng, out_dim, 3),
    )
    @test k isa LatentFactorMOKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k(x1[1], x2[1]) isa Real

    @test kernelmatrix(k, x1, x2) ≈ kernelmatrix(k, collect(x1), collect(x2))
    @test kernelmatrix(k, x1, x1) ≈ kernelmatrix(k, x1)

    @test repr(k) ==
          "Semiparametric Latent Factor Multi-Output Kernel " *
          "(# outputs = $out_dim, # latent processes = 3)"

    # AD test
    function test_slfm(A::AbstractMatrix, x1, x2)
        k = LatentFactorMOKernel(
            [Matern32Kernel(), SqExponentialKernel(), FBMKernel()],
            IndependentMOKernel(GaussianKernel()),
            A,
        )
        return k((x1, 1), (x2, 1))
    end

    a = rand()
    @test all(
        FiniteDifferences.j′vp(FDM, test_slfm, a, k.A, x1[1][1], x2[1][1]) .≈
        Zygote.pullback(test_slfm, k.A, x1[1][1], x2[1][1])[2](a),
    )
end

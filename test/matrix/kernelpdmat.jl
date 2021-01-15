@testset "kernelpdmat" begin
    rng = MersenneTwister(123456)
    A = rand(rng, 10, 5)
    k = SqExponentialKernel()
    for obsdim in [1, 2]
        @test all(
            Matrix(kernelpdmat(k, A; obsdim=obsdim)) .≈
            Matrix(PDMat(kernelmatrix(k, A; obsdim=obsdim))),
        )
        # @test_throws ErrorException kernelpdmat(k,ones(100,100),obsdim=obsdim)
    end
end

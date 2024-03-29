@testset "kernelpdmat" begin
    rng = MersenneTwister(123456)
    A = rand(rng, 10, 5)
    vecA = (RowVecs(A), ColVecs(A))
    a = rand(rng, 10)
    k = SqExponentialKernel()
    for obsdim in [1, 2]
        @test all(
            Matrix(kernelpdmat(k, A; obsdim=obsdim)) .≈
            Matrix(PDMat(kernelmatrix(k, A; obsdim=obsdim))),
        )
        @test kernelpdmat(k, vecA[obsdim]) == kernelpdmat(k, A; obsdim=obsdim)
        # @test_throws ErrorException kernelpdmat(k,ones(100,100),obsdim=obsdim)
    end
    @test @test_deprecated(kernelpdmat(k, A)) == kernelpdmat(k, ColVecs(A))
end

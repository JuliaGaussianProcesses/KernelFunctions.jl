@testset "kernelpdmat" begin
    rng = MersenneTwister(123456)
    A = rand(rng, 10, 5)
    vecA = (RowVecs(A), ColVecs(A))
    a = rand(rng, 10)
    k = SqExponentialKernel()
    for obsdim in (1, 2)
        res = kernelmatrix(PDMat, k, A; obsdim=obsdim)
        @test res isa PDMat
        @test Matrix(res) ≈ Matrix(@test_deprecated(kernelpdmat(k, A; obsdim=obsdim)))
        @test Matrix(res) ≈ kernelmatrix(k, A; obsdim=obsdim)

        res2 = kernelmatrix(PDMat, k, vecA[obsdim])
        @test res2 isa PDMat
        @test Matrix(res2) ≈ Matrix(@test_deprecated(kernelpdmat(k, vecA[obsdim])))
        @test Matrix(res2) ≈ Matrix(res)
    end
end

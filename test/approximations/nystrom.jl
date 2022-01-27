@testset "nystrom" begin
    dims = [10, 5]
    X = rand(dims...)
    k = SqExponentialKernel()
    for obsdim in [1, 2]
        @test kernelmatrix(k, X; obsdim=obsdim) ≈
              kernelmatrix(nystrom(k, X, 1.0; obsdim=obsdim))
        @test kernelmatrix(k, X; obsdim=obsdim) ≈
              kernelmatrix(nystrom(k, X, collect(1:dims[obsdim]); obsdim=obsdim))
    end
    r = rand()
    @test @test_deprecated(nystrom(k, X, r)) ≈ nystrom(k, ColVecs(X), r)
    @test @test_deprecated(nystrom(k, X, 2:4)) ≈ nystrom(k, ColVecs(X), 2:4)
end

@testset "maha" begin
    rng = MersenneTwister(123456)
    D_in = 3
    v1 = rand(rng, D_in)
    v2 = rand(rng, D_in)

    U = UpperTriangular(rand(rng, 3, 3))
    P = Matrix(Cholesky(U, 'U', 0))
    @assert isposdef(P)

    k = @test_deprecated MahalanobisKernel(; P=P)
    @test k isa TransformedKernel{SqExponentialKernel,<:LinearTransform}
    @test k.transform.A ≈ sqrt(2) .* U
    @test k(v1, v2) ≈ exp(-sqmahalanobis(v1, v2, P))

    # Standardised tests.
    @testset "ColVecs" begin
        x0 = ColVecs(randn(D_in, 3))
        x1 = ColVecs(randn(D_in, 3))
        x2 = ColVecs(randn(D_in, 2))
        TestUtils.test_interface(k, x0, x1, x2)
    end
    @testset "RowVecs" begin
        x0 = RowVecs(randn(3, D_in))
        x1 = RowVecs(randn(3, D_in))
        x2 = RowVecs(randn(2, D_in))
        TestUtils.test_interface(k, x0, x1, x2)
    end
end

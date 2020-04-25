@testset "transformedkernel" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    s = rand(rng)
    v = rand(rng, 3)
    k = SqExponentialKernel()
    kt = TransformedKernel(k,ScaleTransform(s))
    ktard = TransformedKernel(k,ARDTransform(v))
    @test kt(v1, v2) == transform(k, ScaleTransform(s))(v1, v2)
    @test kt(v1, v2) == transform(k, s)(v1,v2)
    @test kt(v1, v2) ≈ k(s * v1, s * v2) atol=1e-5
    @test ktard(v1, v2) ≈ transform(k, ARDTransform(v))(v1, v2) atol=1e-5
    @test ktard(v1, v2) == transform(k,v)(v1, v2)
    @test ktard(v1, v2) == k(v .* v1, v .* v2)
end

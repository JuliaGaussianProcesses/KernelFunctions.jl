@testset "cosine" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    k = CosineKernel()
    @test eltype(k) == Any
    @test kappa(k, 1.0) ≈ -1.0 atol=1e-5
    @test kappa(k, 2.0) ≈ 1.0 atol=1e-5
    @test kappa(k, 1.5) ≈ 0.0 atol=1e-5
    @test kappa(k,x) ≈ cospi(x) atol=1e-5
    @test k(v1, v2) ≈ cospi(sqrt(sum(abs2.(v1-v2)))) atol=1e-5
    @test repr(k) == "Cosine Kernel"
    test_ADs(CosineKernel)
end

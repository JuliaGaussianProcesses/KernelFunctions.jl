@testset "FBM" begin
    rng = MersenneTwister(42)
    h = 0.3
    k = FBMKernel(h = h)
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    @test k(v1,v2) ≈ (sqeuclidean(v1, zero(v1))^h + sqeuclidean(v2, zero(v2))^h - sqeuclidean(v1-v2, zero(v1-v2))^h)/2 atol=1e-5
    @test repr(k) == "Fractional Brownian Motion Kernel (h = $(h))"


    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    test_ADs(FBMKernel; ADs = [:ReverseDiff])
    @test_broken "Tests failing for kernelmatrix(k, x) for ForwardDiff and Zygote"
    test_params(k, ([h],))
end

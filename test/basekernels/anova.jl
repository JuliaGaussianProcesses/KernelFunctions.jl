@testset "ANOVA" begin
    rng = MersenneTwister(42)
    d = 2.
    k = ANOVAKernel(d = d)
    v1 = rand(rng, 3); v2 = rand(rng, 3)
    @test k(v1, v2) ≈ sum(exp.( - (v1 .- v2).^2 ) .^ 2)
    @test k(1., 2.) ≈ exp(-1.)^2

    # kernelmatrix tests
    m1 = rand(rng, 3, 3)
    m2 = rand(rng, 3, 3)
    Kref = kernelmatrix(k, m1, m1)
    @test kernelmatrix(k, m1) ≈ Kref atol=1e-5
    K = zeros(3, 3)
    kernelmatrix!(K, k, m1, m1)
    @test K ≈ Kref atol=1e-5
    fill!(K, 0)
    kernelmatrix!(K, k, m1)
    @test K ≈ Kref atol=1e-5

    # repr test
    @test repr(k) == "ANOVA Kernel (d = 2.0)"
    test_ADs(ANOVAKernel)
end

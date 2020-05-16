@testset "FBM" begin
    h = 0.3
    k = FBMKernel(h = h)
    v1 = rand(3); v2 = rand(3)
    @test k(v1,v2) ≈ (sqeuclidean(v1, zero(v1))^h + sqeuclidean(v2, zero(v2))^h - sqeuclidean(v1-v2, zero(v1-v2))^h)/2 atol=1e-5

    # kernelmatrix tests
    m1 = rand(3,3)
    m2 = rand(3,3)
    Kref = kernelmatrix(k, m1, m1)
    @test kernelmatrix(k, m1) ≈ Kref atol=1e-5
    K = zeros(3, 3)
    kernelmatrix!(K, k, m1, m1)
    @test K ≈ Kref atol=1e-5
    fill!(K, 0)
    kernelmatrix!(K, k, m1)
    @test K ≈ Kref atol=1e-5

    x1 = rand()
    x2 = rand()
    @test kernelmatrix(k, x1*ones(1,1), x2*ones(1,1))[1] ≈ k(x1, x2) atol=1e-5

    @test repr(k) == "Fractional Brownian Motion Kernel (h = $(h))"
    test_ADs(FBMKernel, ADs = [:ReverseDiff])
    @test_broken "Tests failing for kernelmatrix(k, x) for ForwardDiff and Zygote"
end

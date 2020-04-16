@testset "maha" begin
    P = rand(3,3)
    k = MahalanobisKernel(P)
    @test kappa(k,x) == exp(-x)
    @test k(v1,v2) ≈ exp(-sqmahalanobis(v1,v2, k.P))
    @test kappa(ExponentialKernel(),x) == kappa(k,x)
end

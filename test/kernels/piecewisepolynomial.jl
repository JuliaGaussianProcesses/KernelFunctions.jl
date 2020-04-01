@testset "piecewisepolynomial" begin
    maha = rand(3,3)
    k = PiecewisePolynomialKernel(1,maha)
    @test 1==1 
    # @test kappa(k,x) == exp(-x)
    # @test k(v1,v2) ≈ exp(-sqmahalanobis(v1,v2, k.P))
    # @test kappa(ExponentialKernel(),x) == kappa(k,x)
end

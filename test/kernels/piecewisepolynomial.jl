@testset "piecewisepolynomial" begin
    v1 = rand(3)
    v2 = rand(3)
    m1 = rand(3,4)
    m2 = rand(3,4)
    maha = ones(3,3)
    k = PiecewisePolynomialKernel{3}(maha)

    @test k(v1,v2) ≈ kappa(k, v1, v2) atol=1e-5
    @test typeof(k(v1,v2)) <: Real
    @test size(k(m1,m2)) == (4,4)
    @test size(kerneldiagmatrix(k, m1)) == (4,)
    @test kerneldiagmatrix(k, m1) == ones(4)
    @test_throws ErrorException PiecewisePolynomialKernel{4}(maha)
end

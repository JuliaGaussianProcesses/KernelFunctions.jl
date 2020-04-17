@testset "piecewisepolynomial" begin
    v1 = rand(3)
    v2 = rand(3)
    m1 = rand(3, 4)
    m2 = rand(3, 4)
    maha = ones(3, 3)
    k = PiecewisePolynomialKernel{3}(maha)

    k2 = PiecewisePolynomialKernel(v=3, maha=maha)

    @test k2(v1, v2) ≈ k(v1, v2) atol=1e-5

    @test k(v1, v2) ≈ kappa(k, v1, v2) atol=1e-5
    @test typeof(k(v1, v2)) <: Real
    @test size(k(m1, m2)) == (4, 4)
    @test size(k(m1)) == (4, 4)

    A1 = ones(4, 4)
    kernelmatrix!(A1, k, m1, m2)
    @test A1 ≈ kernelmatrix(k, m1, m2) atol=1e-5

    A2 = ones(4, 4)
    kernelmatrix!(A2, k, m1)
    @test A2 ≈ kernelmatrix(k, m1) atol=1e-5

    @test size(kerneldiagmatrix(k, m1)) == (4,)
    @test kerneldiagmatrix(k, m1) == ones(4)
    A3 = ones(4)
    kerneldiagmatrix!(A3, k, m1)
    @test A3 == kerneldiagmatrix(k, m1)

    @test_throws ErrorException PiecewisePolynomialKernel{4}(maha)
end

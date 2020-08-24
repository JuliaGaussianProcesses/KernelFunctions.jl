@testset "piecewisepolynomial" begin
    v1 = rand(3)
    v2 = rand(3)
    m1 = rand(3, 4)
    m2 = rand(3, 4)
    maha = ones(3, 3)
    v = 3
    k = PiecewisePolynomialKernel{v}(maha)

    k2 = PiecewisePolynomialKernel(v=v, maha=maha)

    @test k2(v1, v2) ≈ k(v1, v2) atol=1e-5

    @test typeof(k(v1, v2)) <: Real
    @test size(kernelmatrix(k, m1, m2)) == (4, 4)
    @test size(kernelmatrix(k, m1)) == (4, 4)

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

    @test repr(k) == "Piecewise Polynomial Kernel (v = $(v), size(maha) = $(size(maha)))"
    # test_ADs(maha-> PiecewisePolynomialKernel(v=2, maha = maha), maha)
    @test_broken "Nothing passes (problem with Mahalanobis distance in Distances)"

    test_params(k, (maha,))
end

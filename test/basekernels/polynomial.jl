@testset "polynomial" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    c = rand(rng)
    @testset "LinearKernel" begin
        k = LinearKernel()
        @test kappa(k, x) ≈ x
        @test k(v1, v2) ≈ dot(v1, v2)
        @test kappa(LinearKernel(), x) == kappa(k, x)
        @test metric(LinearKernel()) == KernelFunctions.DotProduct()
        @test metric(LinearKernel(; c=c)) == KernelFunctions.DotProduct()
        @test repr(k) == "Linear Kernel (c = 0.0)"

        # Errors.
        @test_throws ArgumentError LinearKernel(; c=-0.5)

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(x -> LinearKernel(; c=x[1]), [c])
        test_params(LinearKernel(; c=c), ([c],))
    end
    @testset "PolynomialKernel" begin
        k = PolynomialKernel()
        @test kappa(k, x) ≈ x^2
        @test k(v1, v2) ≈ dot(v1, v2)^2
        @test kappa(PolynomialKernel(), x) == kappa(k, x)
        @test repr(k) == "Polynomial Kernel (c = 0.0, degree = 2)"

        # Coherence tests.
        @test kappa(PolynomialKernel(; degree=1, c=c), x) ≈ kappa(LinearKernel(; c=c), x)
        @test metric(PolynomialKernel()) == KernelFunctions.DotProduct()
        @test metric(PolynomialKernel(; degree=3)) == KernelFunctions.DotProduct()
        @test metric(PolynomialKernel(; degree=3, c=c)) == KernelFunctions.DotProduct()

        # Errors.
        @test_throws ArgumentError PolynomialKernel(; degree=0)
        @test_throws ArgumentError PolynomialKernel(; c=-0.5)

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(x -> PolynomialKernel(; c=x[1]), [c])
        test_params(PolynomialKernel(; c=c), ([c],))
    end
end

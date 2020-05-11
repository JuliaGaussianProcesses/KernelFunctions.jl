@testset "polynomial" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    c = randn(rng)
    @testset "LinearKernel" begin
        k = LinearKernel()
        @test kappa(k,x) ≈ x
        @test k(v1,v2) ≈ dot(v1,v2)
        @test kappa(LinearKernel(),x) == kappa(k,x)
        @test metric(LinearKernel()) == KernelFunctions.DotProduct()
        @test metric(LinearKernel(c=2.0)) == KernelFunctions.DotProduct()
        @test repr(k) == "Linear Kernel (c = 0.0)"
    end
    @testset "PolynomialKernel" begin
        k = PolynomialKernel()
        @test kappa(k,x) ≈ x^2
        @test k(v1,v2) ≈ dot(v1,v2)^2
        @test kappa(PolynomialKernel(),x) == kappa(k,x)
        @test repr(k) == "Polynomial Kernel (c = 0.0, d = 2.0)"
        #Coherence test
        @test kappa(PolynomialKernel(d=1.0,c=c),x) ≈ kappa(LinearKernel(c=c),x)
        @test metric(PolynomialKernel()) == KernelFunctions.DotProduct()
        @test metric(PolynomialKernel(d=3.0)) == KernelFunctions.DotProduct()
        @test metric(PolynomialKernel(d=3.0,c=2.0)) == KernelFunctions.DotProduct()
    end
end

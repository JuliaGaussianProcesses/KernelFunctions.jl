@testset "transformedkernel" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    s = rand(rng)
    v = rand(rng, 3)
    k = SqExponentialKernel()
    kt = TransformedKernel(k,ScaleTransform(s))
    ktard = TransformedKernel(k,ARDTransform(v))
    @test kappa(kt,v1,v2) == kappa(transform(k,ScaleTransform(s)),v1,v2)
    @test kappa(kt,v1,v2) == kappa(transform(k,s),v1,v2)
    @test kappa(kt,v1,v2) ≈ kappa(k,s*v1,s*v2) atol=1e-5
    @test kappa(ktard,v1,v2) ≈ kappa(transform(k,ARDTransform(v)),v1,v2) atol=1e-5
    @test kappa(ktard,v1,v2) == kappa(transform(k,v),v1,v2)
    @test kappa(ktard,v1,v2) == kappa(k,v.*v1,v.*v2)
    @test KernelFunctions.metric(kt) == KernelFunctions.metric(k)
end

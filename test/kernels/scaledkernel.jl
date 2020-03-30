@testset "scaledkernel" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    s = rand(rng)

    k = SqExponentialKernel()
    ks = ScaledKernel(k,s)
    @test kappa(ks,x) == s*kappa(k,x)
    @test kappa(ks,x) == kappa(s*k,x)
end

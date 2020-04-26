@testset "kernelsum" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k3 = RationalQuadraticKernel()
    X = rand(rng, 2,2)

    w = [2.0,0.5]
    k = KernelSum([k1,k2],w)
    ks1 = 2.0*k1
    ks2 = 0.5*k2
    @test length(k) == 2
    @test k(v1, v2) == (2.0 * k1 + 0.5 * k2)(v1, v2)
    @test (k + k3)(v1,v2) ≈ (k3 + k)(v1, v2)
    @test (k1 + k2)(v1, v2) == KernelSum([k1, k2])(v1, v2)
    @test (k + ks1)(v1, v2) ≈ (ks1 + k)(v1, v2)
    @test (k + k)(v1, v2) == KernelSum([k1, k2, k1, k2], vcat(w, w))(v1, v2)
end

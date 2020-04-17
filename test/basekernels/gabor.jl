@testset "Gabor" begin
    v1 = rand(3); v2 = rand(3)
    ell = abs(rand())
    p = abs(rand())
    k = GaborKernel(ell=ell, p=p)
    @test k.ell ≈ ell atol=1e-5
    @test k.p ≈ p atol=1e-5
    @test kappa(k,v1,v2) ≈ exp(-sqeuclidean(v1,v2) ./(k.ell.^2))*cospi(euclidean(v1,v2)./ k.p) atol=1e-5
    @test kappa(k,v1,v2) ≈ kappa(transform(SqExponentialKernel(), 1/k.ell),v1,v2)*kappa(transform(CosineKernel(), 1/k.p), v1,v2) atol=1e-5

    k = GaborKernel()
    @test k.ell ≈ 1.0 atol=1e-5
    @test k.p ≈ 1.0 atol=1e-5
end

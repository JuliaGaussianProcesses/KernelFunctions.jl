using KernelFunctions
using Test
using Flux

@testset "Params" begin
    ν = 2.0; c = 3.0; d = 2.0; γ = 2.0; α = 2.5
    kc = ConstantKernel(c=c)
    @test all(params(kc) .== params([c]))
    km = MaternKernel(ν=ν)
    @test all(params(km) .== params([ν]))
    kl = LinearKernel(c=c)
    @test all(params(kl) .== params([c]))
    kge = GammaExponentialKernel(γ=γ)
    @test all(params(kge) .== params([γ]))
    kgr = GammaRationalQuadraticKernel(γ=γ, α=α)
    @test all(params(kgr) .== params([α], [γ]))
    kp = PolynomialKernel(c=c, d=d)
    @test all(params(kp) .== params([d], [c]))
    kr = RationalQuadraticKernel(α=α)
    @test all(params(kr) .== params([α]))

    k = km + kc
    @test all(params(k) .== params([k.weights], km, kc))

    k = km * kc
    @test all(params(k) .== params(km, kc))

    s = 2.0
    k = transform(km, s)
    @test all(params(k) .== params([s], km))

    v = [2.0]
    k = transform(kc, v)
    @test all(params(k) .== params(v, kc))

    P = rand(3, 2)
    k = transform(km,LowRankTransform(P))
    @test all(params(k) .== params(P, km))

    k = transform(km, LowRankTransform(P) ∘ ScaleTransform(s))
    @test all(params(k) .== params([s], P, km))

    c = Chain(Dense(3, 2))
    k = transform(km, FunctionTransform(c))
    @test all(params(k) .== params(c, km))

end

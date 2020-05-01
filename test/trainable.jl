@testset "trainable" begin
    ν = 2.0; c = 3.0; d = 2.0; γ = 2.0; α = 2.5; h = 0.5; r = rand(3)

    function test_params(kernel, reference)
        params_kernel = params(kernel)
        params_reference = params(reference)

        @test length(params_kernel) == length(params_reference)
        @test all(p == q for (p, q) in zip(params_kernel, params_reference))
    end

    kc = ConstantKernel(c=c)
    test_params(kc, ([c],))

    kfbm = FBMKernel(h = h)
    test_params(kfbm, ([h],))

    kge = GammaExponentialKernel(γ=γ)
    test_params(kge, ([γ],))

    kgr = GammaRationalQuadraticKernel(γ=γ, α=α)
    test_params(kgr, ([α], [γ]))

    kl = LinearKernel(c=c)
    test_params(kl, ([c],))

    km = MaternKernel(ν=ν)
    test_params(km, ([ν],))

    kp = PolynomialKernel(c=c, d=d)
    test_params(kp, ([d], [c]))

    kpe = PeriodicKernel(r = r)
    test_params(kpe, (r,))

    kr = RationalQuadraticKernel(α=α)
    test_params(kr, ([α],))

    k = km + kc
    test_params(k, (k.weights, km, kc))

    k = km * kc
    test_params(k, (km, kc))

    s = 2.0
    k = transform(km, s)
    test_params(k, ([s], km))

    v = [2.0]
    k = transform(kc, v)
    test_params(k, (v, kc))

    P = rand(3, 2)
    k = transform(km, LinearTransform(P))
    test_params(k, (P, km))

    k = transform(km, LinearTransform(P) ∘ ScaleTransform(s))
    test_params(k, ([s], P, km))

    c = Chain(Dense(3, 2))
    k = transform(km, FunctionTransform(c))
    test_params(k, (c, km))
end

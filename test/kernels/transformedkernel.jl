@testset "transformedkernel" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    s = rand(rng)
    v = rand(rng, 3)
    P = rand(rng, 3, 2)
    k = SqExponentialKernel()
    kt = TransformedKernel(k, ScaleTransform(s))
    ktard = TransformedKernel(k, ARDTransform(v))
    @test kt(v1, v2) == (k ∘ ScaleTransform(s))(v1, v2)
    @test kt(v1, v2) ≈ k(s * v1, s * v2) atol = 1e-5
    @test ktard(v1, v2) == (k ∘ ARDTransform(v))(v1, v2)
    @test ktard(v1, v2) == k(v .* v1, v .* v2)
    @test (k ∘ LinearTransform(P') ∘ ScaleTransform(s))(v1, v2) ==
          ((k ∘ LinearTransform(P')) ∘ ScaleTransform(s))(v1, v2) ==
          (k ∘ (LinearTransform(P') ∘ ScaleTransform(s)))(v1, v2)

    @test repr(kt) == repr(k) * "\n\t- " * repr(ScaleTransform(s))

    TestUtils.test_interface(k, Float64)
    test_ADs(x -> SqExponentialKernel() ∘ ScaleTransform(x[1]), rand(1))

    # Test implicit gradients
    @testset "Implicit gradients" begin
        k = SqExponentialKernel() ∘ ScaleTransform(2.0)
        ps = Flux.params(k)
        X = rand(10, 1)
        x = vec(X)
        A = rand(10, 10)
        # Implicit
        g1 = Flux.gradient(ps) do
            tr(kernelmatrix(k, X; obsdim=1) * A)
        end
        # Explicit
        g2 = Flux.gradient(k) do k
            tr(kernelmatrix(k, X; obsdim=1) * A)
        end

        # Implicit for a vector
        g3 = Flux.gradient(ps) do
            tr(kernelmatrix(k, x) * A)
        end
        @test g1[first(ps)] ≈ first(g2).transform.s
        @test g1[first(ps)] ≈ g3[first(ps)]
    end

    @testset "Parameters" begin
        k = ConstantKernel(; c=rand(rng))
        c = Chain(Dense(3, 2))

        test_params(k ∘ ScaleTransform(s), (k, [s]))
        test_params(k ∘ ARDTransform(v), (k, v))
        test_params(k ∘ LinearTransform(P), (k, P))
        test_params(k ∘ LinearTransform(P) ∘ ScaleTransform(s), (k, [s], P))
        test_params(k ∘ FunctionTransform(c), (k, c))
    end
end

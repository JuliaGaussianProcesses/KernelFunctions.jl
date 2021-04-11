@testset "transformedkernel" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    s = rand(rng)
    s2 = rand(rng)
    v = rand(rng, 3)
    k = SqExponentialKernel()
    kt = TransformedKernel(k, ScaleTransform(s))
    ktard = TransformedKernel(k, ARDTransform(v))
    @test kt(v1, v2) == transform(k, ScaleTransform(s))(v1, v2)
    @test kt(v1, v2) == transform(k, s)(v1, v2)
    @test kt(v1, v2) == (k ∘ ScaleTransform(s))(v1, v2)
    @test kt(v1, v2) ≈ k(s * v1, s * v2) atol = 1e-5
    @test ktard(v1, v2) ≈ transform(k, ARDTransform(v))(v1, v2) atol = 1e-5
    @test ktard(v1, v2) == (k ∘ ARDTransform(v))(v1, v2)
    @test ktard(v1, v2) == transform(k, v)(v1, v2)
    @test ktard(v1, v2) == k(v .* v1, v .* v2)
    @test transform(kt, s2)(v1, v2) ≈ kt(s2 * v1, s2 * v2)
    @test KernelFunctions.kernel(kt) == k
    @test repr(kt) == repr(k) * "\n\t- " * repr(ScaleTransform(s))

    TestUtils.test_interface(k, Float64)
    test_ADs(x -> transform(SqExponentialKernel(), x[1]), rand(1))# ADs = [:ForwardDiff, :ReverseDiff])
    # Test implicit gradients
    @testset "Implicit gradients" begin
        k = transform(SqExponentialKernel(), 2.0)
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

    P = rand(3, 2)
    c = Chain(Dense(3, 2))

    test_params(transform(k, s), (k, [s]))
    test_params(transform(k, v), (k, v))
    test_params(transform(k, LinearTransform(P)), (k, P))
    test_params(transform(k, LinearTransform(P) ∘ ScaleTransform(s)), (k, [s], P))
    test_params(transform(k, FunctionTransform(c)), (k, c))

    @test (k ∘ (LinearTransform(P') ∘ ScaleTransform(s)))(v1, v2) ==
          ((k ∘ LinearTransform(P')) ∘ ScaleTransform(s))(v1, v2)
    test_params(k ∘ LinearTransform(P), (P, k))
    test_params(k ∘ LinearTransform(P) ∘ ScaleTransform(s), ([s], P, k))
    test_params(k ∘ FunctionTransform(c), (c, k))
end

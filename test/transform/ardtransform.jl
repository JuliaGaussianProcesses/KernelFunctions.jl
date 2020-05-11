@testset "ardtransform" begin
    rng = MersenneTwister(123456)

    @testset "Real input" begin
        v = randn(rng, 1)
        t = ARDTransform(v)

        x = randn(rng, 4)
        x′ = map(t, x)

        @test all([t(x[n]) == v[1] * x[n] for n in eachindex(x)])
        @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])
    end

    @testset "Vector input" begin
        D = 3
        v = randn(rng, D)
        t = ARDTransform(v)

        XC = ColVecs(randn(rng, D, 7))
        XR = RowVecs(randn(rng, 2, D))

        @testset "$(typeof(x))" for x in [XC, XR]
            x′ = map(t, x)
            @test all([t(x[n]) == v .* x[n] for n in eachindex(x)])
            @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])
        end
    end

    # Check that setting the parameters works.
    D = 4
    t = ARDTransform(randn(rng, D))
    v = randn(rng, D)
    KernelFunctions.set!(t, v)
    @test all(t.v .== v)

    # Check ARD constructor with constant fill.
    s = randn(rng)
    @test ARDTransform(s, D).v == ARDTransform(s * ones(D)).v

    @test_throws DimensionMismatch map(t, ColVecs(randn(rng, D + 1, 3)))

    @test repr(t) == "ARD Transform (dims: $D)"
end

@testset "lowranktransform" begin
    rng = MersenneTwister(123546)

    @testset "Real inputs" begin
        P = randn(rng, 3, 1)
        t = LowRankTransform(P)

        x = randn(rng, 4)
        x′ = map(t, x)

        @test all([t(x[n]) ≈ P * x[n] for n in eachindex(x)])
        @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])
    end

    @testset "Vector inputs" begin
        Din = 3
        Dout = 4
        P = randn(rng, Dout, Din)
        t = LowRankTransform(P)

        x_cols = ColVecs(randn(rng, Din, 8))
        x_rows = RowVecs(randn(rng, 9, Din))

        @testset "$(typeof(x))" for x in [x_cols, x_rows]
            x′ = map(t, x)
            @test all([t(x[n]) ≈ P * x[n] for n in eachindex(x)])
            @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])
        end
    end

    Din = 2
    Dout = 5
    P = randn(rng, Dout, Din)
    t = LowRankTransform(P)

    P2 = randn(rng, Dout, Din)
    KernelFunctions.set!(t, P2)
    @test t.proj == P2
    @test_throws AssertionError KernelFunctions.set!(t, rand(rng, Din + 1, Dout))

    @test_throws DimensionMismatch map(t, ColVecs(randn(rng, Din + 1, Dout)))

    @test repr(t) == "Low Rank Transform (size(P) = ($Dout, $Din))"
end

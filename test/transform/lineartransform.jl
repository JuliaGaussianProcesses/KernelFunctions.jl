@testset "lineartransform" begin
    rng = MersenneTwister(123546)

    @testset "Real inputs" begin
        P = randn(rng, 3, 1)
        t = LinearTransform(P)

        x = randn(rng, 4)
        x′ = map(t, x)

        @test all([t(x[n]) ≈ P * x[n] for n in eachindex(x)])
        @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])
    end

    @testset "Vector inputs" begin
        Din = 3
        Dout = 4
        P = randn(rng, Dout, Din)
        t = LinearTransform(P)

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
    t = LinearTransform(P)

    P2 = randn(rng, Dout, Din)
    KernelFunctions.set!(t, P2)
    @test t.A == P2
    @test_throws ErrorException KernelFunctions.set!(t, rand(rng, Din + 1, Dout))

    @test_throws DimensionMismatch map(t, ColVecs(randn(rng, Din + 1, Dout)))

    @test repr(t) == "Linear transform (size(A) = ($Dout, $Din))"
end

@testset "functiontransform" begin
    rng = MersenneTwister(123546)

    @testset "Real input" begin
        t = FunctionTransform(sin)

        x = randn(rng, 4)
        x′ = map(t, x)

        @test all([t(x[n]) == sin(x[n]) for n in eachindex(x)])
        @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])
    end

    @testset "Vector input" begin
        f = x -> sin.(x)
        t = FunctionTransform(f)

        x_cols = ColVecs(randn(rng, 4, 7))
        x_rows = RowVecs(randn(rng, 6, 3))

        @testset "$(typeof(x))" for x in [x_cols, x_rows]
            x′ = map(t, x)
            @test all([t(x[n]) ≈ f(x[n]) for n in eachindex(x)])
            @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])
        end
    end

    @test repr(FunctionTransform(sin)) == "Function Transform: $(sin)"
end

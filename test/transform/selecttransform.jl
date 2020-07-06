@testset "selecttransform" begin
    rng = MersenneTwister(123456)

    select = [1, 3, 5]
    t = SelectTransform(select)

    x_vecs = [randn(rng, maximum(select)) for _ in 1:3]
    x_cols = ColVecs(randn(rng, maximum(select), 6))
    x_rows = RowVecs(randn(rng, 4, maximum(select)))

    @testset "$(typeof(x))" for x in [x_vecs, x_cols, x_rows]
        x′ = map(t, x)
        @test all([t(x[n]) == x[n][select] for n in eachindex(x)])
        @test all([t(x[n]) == x′[n] for n in eachindex(x)])
    end

    select2 = [2, 3, 5]
    KernelFunctions.set!(t, select2)
    @test t.select == select2

    @test repr(t) == "Select Transform (dims: $(select2))"
    test_ADs(()->transform(SEKernel(), SelectTransform([1,2])))
end

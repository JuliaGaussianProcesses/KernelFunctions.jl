@testset "selecttransform" begin
    rng = MersenneTwister(123456)

    select = [1, 3, 5]
    t = SelectTransform(select)

    x_vecs = [randn(rng, maximum(select)) for _ in 1:3]
    x_cols = ColVecs(randn(rng, maximum(select), 6))
    x_rows = RowVecs(randn(rng, 4, maximum(select)))

    Xs = [x_vecs, x_cols, x_rows]

    @testset "$(typeof(x))" for x in Xs
        x′ = map(t, x)
        @test all([t(x[n]) == x[n][select] for n in eachindex(x)])
        @test all([t(x[n]) == x′[n] for n in eachindex(x)])
    end

    symbols = [:a, :b, :c, :d, :e]
    select_symbols = [:a, :c, :e]

    ts = SelectTransform(select_symbols)

    a_vecs = map(x->AxisArray(x, col=symbols), x_vecs)
    a_cols = ColVecs(AxisArray(x_cols.X, col=symbols, index=(1:6)))
    a_rows = RowVecs(AxisArray(x_rows.X, index=(1:4), col=symbols))

    As = [a_vecs, a_cols, a_rows]

    @testset "$(typeof(a))" for (a, x) in zip(As, Xs)
        a′ = map(ts, a)
        x′ = map(t, x)
        @test a′ == x′ 
    end

    select2 = [2, 3, 5]
    KernelFunctions.set!(t, select2)
    @test t.select == select2

    select_symbols2 = [:b, :c, :e]
    KernelFunctions.set!(ts, select_symbols2)
    @test ts.select == select_symbols2

    @test repr(t) == "Select Transform (dims: $(select2))"
    @test repr(ts) == "Select Transform (dims: $(select_symbols2))"

    test_ADs(()->transform(SEKernel(), SelectTransform([1,2])))
end

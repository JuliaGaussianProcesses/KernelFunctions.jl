@testset "selecttransform" begin
    rng = MersenneTwister(123456)

    select = [1, 3, 5]
    t = SelectTransform(select)

    x_vecs = [randn(rng, maximum(select)) for _ in 1:3]
    x_cols = ColVecs(randn(rng, maximum(select), 6))
    x_rows = RowVecs(randn(rng, 4, maximum(select)))
    x_string = ColVecs([randstring(rng) for _ in 1:maximum(select), _ in 1:5])

    Xs = [x_vecs, x_cols, x_rows, x_string]

    @testset "$(typeof(x))" for x in Xs
        x′ = map(t, x)
        @test all([t(x[n]) == x[n][select] for n in eachindex(x)])
        @test all([t(x[n]) == x′[n] for n in eachindex(x)])
    end

    symbols = [:a, :b, :c, :d, :e]
    select_symbols = [:a, :c, :e]

    ts = SelectTransform(select_symbols)

    a_vecs = map(x -> AxisArray(x; col=symbols), x_vecs)
    a_cols = ColVecs(AxisArray(x_cols.X; col=symbols, index=(1:6)))
    a_rows = RowVecs(AxisArray(x_rows.X; index=(1:4), col=symbols))
    a_string = ColVecs(AxisArray(x_string.X; col=symbols, index=1:5))

    As = [a_vecs, a_cols, a_rows, a_string]

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

    test_ADs(() -> SEKernel() ∘ SelectTransform([1, 2]))
    test_interface_ad_perf(
        _ -> SEKernel(),
        nothing,
        StableRNG(123456),
        [ColVecs{Float64,Matrix{Float64}}, RowVecs{Float64,Matrix{Float64}}],
    )

    X = randn(rng, (4, 3))
    A = AxisArray(X; row=[:a, :b, :c, :d], col=[:x, :y, :z])
    Y = randn(rng, (4, 2))
    B = AxisArray(Y; row=[:a, :b, :c, :d], col=[:v, :w])
    Z = randn(rng, (2, 3))
    C = AxisArray(Z; row=[:e, :f], col=[:x, :y, :z])

    tx_row = SEKernel() ∘ SelectTransform([1, 2, 4])
    ta_row = SEKernel() ∘ SelectTransform([:a, :b, :d])
    tx_col = SEKernel() ∘ SelectTransform([1, 3])
    ta_col = SEKernel() ∘ SelectTransform([:x, :z])

    @test kernelmatrix(tx_row, X; obsdim=2) ≈ kernelmatrix(ta_row, A; obsdim=2)
    @test kernelmatrix(tx_col, X; obsdim=1) ≈ kernelmatrix(ta_col, A; obsdim=1)

    @test kernelmatrix(tx_row, X, Y; obsdim=2) ≈ kernelmatrix(ta_row, A, B; obsdim=2)
    @test kernelmatrix(tx_col, X, Z; obsdim=1) ≈ kernelmatrix(ta_col, A, C; obsdim=1)

    @testset "$(AD)" for AD in [:ForwardDiff]
        gx = gradient(AD, X) do x
            testfunction(tx_row, x, 2)
        end
        ga = gradient(AD, A) do a
            testfunction(ta_row, a, 2)
        end
        @test gx ≈ ga
        gx = gradient(AD, X) do x
            testfunction(tx_col, x, 1)
        end
        ga = gradient(AD, A) do a
            testfunction(ta_col, a, 1)
        end
        @test gx ≈ ga
        gx = gradient(AD, X) do x
            testfunction(tx_row, x, Y, 2)
        end
        ga = gradient(AD, A) do a
            testfunction(ta_row, a, B, 2)
        end
        @test gx ≈ ga
        gx = gradient(AD, X) do x
            testfunction(tx_col, x, Z, 1)
        end
        ga = gradient(AD, A) do a
            testfunction(ta_col, a, C, 1)
        end
        @test gx ≈ ga
    end

    @testset "$(AD)" for AD in [:ReverseDiff, :Zygote]
        @test_broken let
            gx = gradient(AD, X) do x
                testfunction(tx_row, x, 2)
            end
            ga = gradient(AD, A) do a
                testfunction(ta_row, a, 2)
            end
            gx ≈ ga
        end
        @test_broken let
            gx = gradient(AD, X) do x
                testfunction(tx_col, x, 1)
            end
            ga = gradient(AD, A) do a
                testfunction(ta_col, a, 1)
            end
            gx ≈ ga
        end
        @test_broken let
            gx = gradient(AD, X) do x
                testfunction(tx_row, x, Y, 2)
            end
            ga = gradient(AD, A) do a
                testfunction(ta_row, a, B, 2)
            end
            gx ≈ ga
        end
        @test_broken let
            gx = gradient(AD, X) do x
                testfunction(tx_col, x, Z, 1)
            end
            ga = gradient(AD, A) do a
                testfunction(ta_col, a, C, 1)
            end
            gx ≈ ga
        end
    end

    @testset "single-index" begin
        t = SelectTransform(4)
        @testset "$(name)" for (name, x) in [
            ("Vector{<:Vector}", [randn(6) for _ in 1:3]),
            ("ColVecs", ColVecs(randn(5, 10))),
            ("RowVecs", RowVecs(randn(11, 4))),
            ("ColVecs{String}", ColVecs([randstring() for _ in 1:6, _ in 1:5])),
        ]
            @test KernelFunctions._map(t, x) isa AbstractVector
        end
    end
end

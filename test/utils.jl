@testset "utils" begin
    using KernelFunctions: vec_of_vecs, ColVecs, RowVecs
    rng, N, D = MersenneTwister(123456), 10, 4
    x, X = randn(rng, N), randn(rng, D, N)
    @testset "VecOfVecs" begin
        @test vec_of_vecs(X, obsdim = 2) == ColVecs(X)
        @test vec_of_vecs(X, obsdim = 1) == RowVecs(X)
    end
    # Test Matrix data sets.
    @testset "ColVecs" begin
        DX = ColVecs(X)
        @test DX == DX
        @test size(DX) == (N,)
        @test length(DX) == N
        @test getindex(DX, 5) isa AbstractVector
        @test getindex(DX, 5) == X[:, 5]
        @test getindex(DX, 1:2:6) isa ColVecs
        @test getindex(DX, 1:2:6) == ColVecs(X[:, 1:2:6])
        @test getindex(DX, :) == ColVecs(X)
        @test eachindex(DX) == 1:N
        @test first(DX) == X[:, 1]

        Y = randn(rng, D, N + 1)
        DY = ColVecs(Y)
        @test pairwise(SqEuclidean(), DX) ≈ pairwise(SqEuclidean(), X; dims=2)
        @test pairwise(SqEuclidean(), DX, DY) ≈ pairwise(SqEuclidean(), X, Y; dims=2)
        K = zeros(N, N)
        pairwise!(K, SqEuclidean(), DX)
        @test K ≈ pairwise(SqEuclidean(), X; dims=2)
        K = zeros(N, N + 1)
        pairwise!(K, SqEuclidean(), DX, DY)
        @test K ≈ pairwise(SqEuclidean(), X, Y; dims=2)

        let
            @test Zygote.pullback(ColVecs, X)[1] == DX
            DX, back = Zygote.pullback(ColVecs, X)
            @test back((X=ones(size(X)),))[1] == ones(size(X))

            @test Zygote.pullback(DX->DX.X, DX)[1] == X
            X_, back = Zygote.pullback(DX->DX.X, DX)
            @test back(ones(size(X)))[1].X == ones(size(X))
        end
    end
    @testset "RowVecs" begin
        DX = RowVecs(X)
        @test DX == DX
        @test size(DX) == (D,)
        @test length(DX) == D
        @test getindex(DX, 2) isa AbstractVector
        @test getindex(DX, 2) == X[2, :]
        @test getindex(DX, 1:3) isa RowVecs
        @test getindex(DX, 1:3) == RowVecs(X[1:3, :])
        @test getindex(DX, :) == RowVecs(X)
        @test eachindex(DX) == 1:D
        @test first(DX) == X[1, :]

        Y = randn(rng, D + 1, N)
        DY = RowVecs(Y)
        @test pairwise(SqEuclidean(), DX) ≈ pairwise(SqEuclidean(), X; dims=1)
        @test pairwise(SqEuclidean(), DX, DY) ≈ pairwise(SqEuclidean(), X, Y; dims=1)
        K = zeros(D, D)
        pairwise!(K, SqEuclidean(), DX)
        @test K ≈ pairwise(SqEuclidean(), X; dims=1)
        K = zeros(D, D + 1)
        pairwise!(K, SqEuclidean(), DX, DY)
        @test K ≈ pairwise(SqEuclidean(), X, Y; dims=1)

        let
            @test Zygote.pullback(RowVecs, X)[1] == DX
            DX, back = Zygote.pullback(RowVecs, X)
            @test back((X=ones(size(X)),))[1] == ones(size(X))

            @test Zygote.pullback(DX->DX.X, DX)[1] == X
            X_, back = Zygote.pullback(DX->DX.X, DX)
            @test back(ones(size(X)))[1].X == ones(size(X))
        end
    end
end

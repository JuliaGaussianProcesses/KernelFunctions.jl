@testset "utils" begin
    using KernelFunctions: vec_of_vecs, ColVecs, RowVecs
    rng, N, D = MersenneTwister(123456), 10, 4
    x = randn(rng, N)
    X = randn(rng, D, N)
    v = randn(rng, D)
    w = randn(rng, N)

    @testset "VecOfVecs" begin
        @test vec_of_vecs(X; obsdim=2) == ColVecs(X)
        @test vec_of_vecs(X; obsdim=1) == RowVecs(X)
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
        DX[2] = v
        @test DX[2] == v
        @test X[:, 2] == v

        Y = randn(rng, D, N + 1)
        DY = ColVecs(Y)
        @test KernelFunctions.pairwise(SqEuclidean(), DX) ≈
              pairwise(SqEuclidean(), X; dims=2)
        @test KernelFunctions.pairwise(SqEuclidean(), DX, DY) ≈
              pairwise(SqEuclidean(), X, Y; dims=2)
        @test vcat(DX, DY) isa ColVecs
        @test vcat(DX, DY).X == hcat(X, Y)
        K = zeros(N, N)
        KernelFunctions.pairwise!(K, SqEuclidean(), DX)
        @test K ≈ pairwise(SqEuclidean(), X; dims=2)
        K = zeros(N, N + 1)
        KernelFunctions.pairwise!(K, SqEuclidean(), DX, DY)
        @test K ≈ pairwise(SqEuclidean(), X, Y; dims=2)

        let
            @test Zygote.pullback(ColVecs, X)[1] == DX
            DX, back = Zygote.pullback(ColVecs, X)
            @test back((X=ones(size(X)),))[1] == ones(size(X))

            @test Zygote.pullback(DX -> DX.X, DX)[1] == X
            X_, back = Zygote.pullback(DX -> DX.X, DX)
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
        DX[2] = w
        @test DX[2] == w
        @test X[2, :] == w

        Y = randn(rng, D + 1, N)
        DY = RowVecs(Y)
        @test KernelFunctions.pairwise(SqEuclidean(), DX) ≈
              pairwise(SqEuclidean(), X; dims=1)
        @test KernelFunctions.pairwise(SqEuclidean(), DX, DY) ≈
              pairwise(SqEuclidean(), X, Y; dims=1)
        @test vcat(DX, DY) isa RowVecs
        @test vcat(DX, DY).X == vcat(X, Y)
        K = zeros(D, D)
        KernelFunctions.pairwise!(K, SqEuclidean(), DX)
        @test K ≈ pairwise(SqEuclidean(), X; dims=1)
        K = zeros(D, D + 1)
        KernelFunctions.pairwise!(K, SqEuclidean(), DX, DY)
        @test K ≈ pairwise(SqEuclidean(), X, Y; dims=1)

        let
            @test Zygote.pullback(RowVecs, X)[1] == DX
            DX, back = Zygote.pullback(RowVecs, X)
            @test back((X=ones(size(X)),))[1] == ones(size(X))

            @test Zygote.pullback(DX -> DX.X, DX)[1] == X
            X_, back = Zygote.pullback(DX -> DX.X, DX)
            @test back(ones(size(X)))[1].X == ones(size(X))
        end
    end
    @testset "ColVecs + RowVecs" begin
        x_colvecs = ColVecs(randn(3, 5))
        x_rowvecs = RowVecs(randn(7, 3))

        @test isapprox(
            pairwise(SqEuclidean(), x_colvecs, x_rowvecs),
            pairwise(SqEuclidean(), collect(x_colvecs), collect(x_rowvecs)),
        )
        @test isapprox(
            pairwise(SqEuclidean(), x_rowvecs, x_colvecs),
            pairwise(SqEuclidean(), collect(x_rowvecs), collect(x_colvecs)),
        )
    end
    @testset "input checks" begin
        D = 3
        D⁻ = 2
        N1 = 2
        N2 = 3
        x = [rand(rng, D) for _ in 1:N1]
        x⁻ = [rand(rng, D⁻) for _ in 1:N1]
        y = [rand(rng, D) for _ in 1:N2]
        xx = [rand(rng, D, D) for _ in 1:N1]
        xx⁻ = [rand(rng, D, D⁻) for _ in 1:N1]
        yy = [rand(rng, D, D) for _ in 1:N2]

        @test KernelFunctions.dim("string") == 0
        @test KernelFunctions.dim(["string", "string2"]) == 0
        @test KernelFunctions.dim(rand(rng, 4)) == 1
        @test KernelFunctions.dim(x) == D

        @test_nowarn KernelFunctions.validate_inplace_dims(zeros(N1, N2), x, y)
        @test_throws DimensionMismatch KernelFunctions.validate_inplace_dims(
            zeros(N1, N1), x, y
        )
        @test_throws DimensionMismatch KernelFunctions.validate_inplace_dims(
            zeros(N1, N2), x⁻, y
        )
        @test_nowarn KernelFunctions.validate_inplace_dims(zeros(N1, N1), x)
        @test_nowarn KernelFunctions.validate_inplace_dims(zeros(N1), x)
        @test_throws DimensionMismatch KernelFunctions.validate_inplace_dims(zeros(N2), x)

        @test_nowarn KernelFunctions.validate_inputs(x, y)
        @test_throws DimensionMismatch KernelFunctions.validate_inputs(x⁻, y)

        @test_nowarn KernelFunctions.validate_inputs(xx, yy)
        @test_nowarn KernelFunctions.validate_inputs(xx⁻, yy)
    end
end

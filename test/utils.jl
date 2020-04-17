@testset "utils" begin
    using KernelFunctions: ColVecs
    rng, N, D = MersenneTwister(123456), 10, 2
    x, X = randn(rng, N), randn(rng, D, N)

    # Test Matrix data sets.
    @testset "ColVecs" begin
        DX = ColVecs(X)
        @test DX == DX
        @test size(DX) == (N,)
        @test length(DX) == N
        @test getindex(DX, 5) isa Vector
        @test getindex(DX, 5) == X[:, 5]
        @test getindex(DX, 1:2:6) isa ColVecs
        @test getindex(DX, 1:2:6) == ColVecs(X[:, 1:2:6])
        @test view(DX, 4) isa AbstractVector
        @test view(DX, 4) == view(X, :, 4)
        @test view(DX, 1:2:4) isa ColVecs
        @test view(DX, 1:2:4) == ColVecs(view(X, :, 1:2:4))
        @test eltype(DX) == Vector{Float64}
        @test eachindex(DX) == 1:N

        let
            @test Zygote.pullback(ColVecs, X)[1] == DX
            DX, back = Zygote.pullback(ColVecs, X)
            @test back((X=ones(size(X)),))[1] == ones(size(X))

            @test Zygote.pullback(DX->DX.X, DX)[1] == X
            X_, back = Zygote.pullback(DX->DX.X, DX)
            @test back(ones(size(X)))[1].X == ones(size(X))
        end
    end
end

@testset "zygote_rules" begin
    @testset "colwise(::Euclidean, X, Y; dims=2)" begin
        rng, D, P = MersenneTwister(123456), 2, 3
        X, Y, D̄ = randn(rng, D, P), randn(rng, D, P), randn(rng, P)
        adjoint_test((X, Y)->colwise(Euclidean(), X, Y), D̄, X, Y)
    end
    @testset "pairwise(::Euclidean, X, Y; dims=2)" begin
        rng, D, P, Q = MersenneTwister(123456), 2, 3, 5
        X, Y, D̄ = randn(rng, D, P), randn(rng, D, Q), randn(rng, P, Q)
        adjoint_test((X, Y)->pairwise(Euclidean(), X, Y; dims=2), D̄, X, Y)
    end
    @testset "pairwise(::Euclidean, X; dims=2)" begin
        rng, D, P = MersenneTwister(123456), 2, 3
        X, D̄ = randn(rng, D, P), randn(rng, P, P)
        adjoint_test(X->pairwise(Euclidean(), X; dims=2), D̄, X)
    end
end

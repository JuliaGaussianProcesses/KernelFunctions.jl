@testset "Chain Rules" begin
    rng = MersenneTwister(123456)
    x = rand(rng, 5)
    y = rand(rng, 5)
    r = rand(rng, 5)
    Q = Matrix(Cholesky(rand(rng, 5, 5), 'U', 0))
    @assert isposdef(Q)

    compare_gradient(:Zygote, [x, y]) do xy
        Euclidean()(xy[1], xy[2])
    end
    compare_gradient(:Zygote, [x, y]) do xy
        SqEuclidean()(xy[1], xy[2])
    end
    compare_gradient(:Zygote, [x, y]) do xy
        KernelFunctions.DotProduct()(xy[1], xy[2])
    end
    compare_gradient(:Zygote, [x, y]) do xy
        KernelFunctions.Delta()(xy[1], xy[2])
    end
    compare_gradient(:Zygote, [x, y]) do xy
        KernelFunctions.Sinus(r)(xy[1], xy[2])
    end
    if VERSION < v"1.6"
        @test_broken "Chain rule of SqMahalanobis is broken in Julia pre-1.6"
    else
        compare_gradient(:Zygote, [Q, x, y]) do Qxy
            SqMahalanobis(Qxy[1])(Qxy[2], Qxy[3])
        end
    end

    @testset "rrules for Sinus(r=$r)" for r in (rand(3),)
        dist = KernelFunctions.Sinus(r)
        @testset "$type" for type in (Vector, SVector{3})
            test_rrule(dist, type(rand(3)), type(rand(3)))
        end
        @testset "$type1, $type2" for type1 in (Matrix, SMatrix{3,2}),
            type2 in (Matrix, SMatrix{3,4})

            test_rrule(Distances.pairwise, dist, type1(rand(3, 2)); fkwargs=(dims=2,))
            test_rrule(
                Distances.pairwise, dist, type1(rand(3, 2)), type2(rand(3, 4));
                fkwargs=(dims=2,)
            )
            test_rrule(Distances.colwise, dist, type1(rand(3, 2)), type1(rand(3, 2)))
        end
    end
end

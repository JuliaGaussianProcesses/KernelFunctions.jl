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
        compare_gradient(:ForwardDiff, [Q, x, y]) do Qxy
            SqMahalanobis(Qxy[1]; skipchecks=true)(Qxy[2], Qxy[3])
        end
    else
        compare_gradient(:Zygote, [Q, x, y]) do Qxy
            SqMahalanobis(Qxy[1])(Qxy[2], Qxy[3])
        end
    end
end

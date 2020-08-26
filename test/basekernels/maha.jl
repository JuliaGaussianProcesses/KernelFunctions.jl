@testset "maha" begin
    rng = MersenneTwister(123456)
    x = 2 * rand(rng)
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    U = UpperTriangular(rand(rng, 3,3))
    P = Matrix(Cholesky(U, 'U', 0))
    @assert isposdef(P)
    k = MahalanobisKernel(P)
  
    @test kappa(k, x) == exp(-x)
    @test k(v1, v2) ≈ exp(-sqmahalanobis(v1, v2, P))
    @test kappa(ExponentialKernel(), x) == kappa(k, x)
    @test repr(k) == "Mahalanobis Kernel (size(P) = $(size(P)))"

    M1, M2 = rand(rng,3,2), rand(rng,3,2)
    fdm = FiniteDifferences.Central(5, 1);
    
    
    function FiniteDifferences.to_vec(dist::SqMahalanobis{Float64})
        return vec(dist.qmat), x -> SqMahalanobis(reshape(x, size(dist.qmat)...))
    end
    
    function test_mahakernel(U::UpperTriangular, v1::AbstractVector, v2::AbstractVector)
        return MahalanobisKernel(Array(U'*U))(v1, v2)
    end
    
    @test_broken all(j′vp(fdm, test_mahakernel, 1, U, v1, v2) .≈
        Zygote.pullback(test_mahakernel, U, v1, v2)[2](1))
    
    function test_sqmaha(U::UpperTriangular, v1::AbstractVector, v2::AbstractVector)
        return SqMahalanobis(Array(U'*U))(v1, v2)
    end
    
    @test_broken all(j′vp(fdm, test_sqmaha, 1, U, v1, v2) .≈ 
    Zygote.pullback(test_sqmaha, U, v1, v2)[2](1))
    
    # test_ADs(U -> MahalanobisKernel(Array(U' * U)), U, ADs=[:Zygote])
    @test_broken "Nothing passes (problem with Mahalanobis distance in Distances)"

    test_params(k, (P,))
end

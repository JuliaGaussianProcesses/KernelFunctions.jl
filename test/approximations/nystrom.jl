@testset "nystrom" begin
    dims = [10, 5]
    X = rand(dims...)
    k = SqExponentialKernel()
    for obsdim in [1, 2]
        Xv = vec_of_vecs(X; obsdim)
        @assert Xv isa Union{ColVecs,RowVecs}
        @test kernelmatrix(k, Xv) ≈ kernelmatrix(nystrom(k, Xv, 1.0))
        @test kernelmatrix(k, Xv) ≈ kernelmatrix(nystrom(k, Xv, collect(1:dims[obsdim])))
    end
    for obsdim in [1, 2]
        @test kernelmatrix(k, X; obsdim=obsdim) ≈
            kernelmatrix(nystrom(k, X, 1.0; obsdim=obsdim))
        @test kernelmatrix(k, X; obsdim=obsdim) ≈
            kernelmatrix(nystrom(k, X, collect(1:dims[obsdim]); obsdim=obsdim))
    end
end

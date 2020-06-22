@testset "transformedkernel" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    s = rand(rng)
    v = rand(rng, 3)
    k = SqExponentialKernel()
    kt = TransformedKernel(k, ScaleTransform(s))
    ktard = TransformedKernel(k, ARDTransform(v))
    @test kt(v1, v2) == transform(k, ScaleTransform(s))(v1, v2)
    @test kt(v1, v2) == transform(k, s)(v1,v2)
    @test kt(v1, v2) ≈ k(s * v1, s * v2) atol=1e-5
    @test ktard(v1, v2) ≈ transform(k, ARDTransform(v))(v1, v2) atol=1e-5
    @test ktard(v1, v2) == transform(k,v)(v1, v2)
    @test ktard(v1, v2) == k(v .* v1, v .* v2)

    @test transform(kt, s) isa TransformedKernel{SqExponentialKernel,ChainTransform{Array{ScaleTransform{Float64},1}}}

    @test transform(k, s) isa TransformedKernel{SqExponentialKernel,ScaleTransform{Float64}}
    @test transform(k, v) isa TransformedKernel{SqExponentialKernel,ARDTransform{Array{Float64,1}}}
    @test transform(k, rand(3, 2)) isa TransformedKernel{SqExponentialKernel,LinearTransform{Array{Float64,2}}}

    @testset "kernelmatrix" begin
        rng = MersenneTwister(123456)

        Nx = 5
        Ny = 4
        D = 3

        k = SqExponentialKernel()
        t = ScaleTransform(randn(rng))
        kt = TransformedKernel(k, t)

        @testset "$(typeof(x))" for (x, y) in [
            (ColVecs(randn(rng, D, Nx)), ColVecs(randn(rng, D, Ny))),
            (RowVecs(randn(rng, Nx, D)), RowVecs(randn(rng, Ny, D))),
        ]
            @test kernelmatrix(kt, x, y) ≈ kernelmatrix(k, map(t, x), map(t, y))

            @test kernelmatrix(kt, x) ≈ kernelmatrix(k, map(t, x))

            @test kerneldiagmatrix(kt, x) ≈ kerneldiagmatrix(k, map(t, x))

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test kernelmatrix!(tmp, kt, x, y) ≈ kernelmatrix(kt, x, y)

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test kernelmatrix!(tmp_square, kt, x) ≈ kernelmatrix(kt, x)

            tmp_diag = Vector{Float64}(undef, length(x))
            @test kerneldiagmatrix!(tmp_diag, kt, x) ≈ kerneldiagmatrix(kt, x)
        end
    end
    test_ADs(x->transform(SqExponentialKernel(), x[1]), rand(1))# ADs = [:ForwardDiff, :ReverseDiff])
end

@testset "scaledkernel" begin
    rng = MersenneTwister(123456)
    x = randn(rng)
    y = randn(rng)
    s = rand(rng) + 1e-3

    k = SqExponentialKernel()
    ks = ScaledKernel(k, s)
    @test ks(x, y) == s * k(x, y)
    @test ks(x, y) == (s * k)(x, y)

    @testset "kernelmatrix" begin
        rng = MersenneTwister(123456)

        Nx = 5
        Ny = 4
        D = 3

        k = SqExponentialKernel()
        s = rand(rng) + 1e-3
        ks = s * k

        @testset "$(typeof(x))" for (x, y) in [
            (ColVecs(randn(rng, D, Nx)), ColVecs(randn(rng, D, Ny))),
            (RowVecs(randn(rng, Nx, D)), RowVecs(randn(rng, Ny, D))),
        ]
            @test kernelmatrix(ks, x, y) ≈ s .* kernelmatrix(k, x, y)

            @test kernelmatrix(ks, x) ≈ s .* kernelmatrix(k, x)

            @test kerneldiagmatrix(ks, x) ≈ s .* kerneldiagmatrix(k, x)

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test_broken kernelmatrix!(tmp, ks, x, y) ≈ kernelmatrix(ks, x, y)

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test_broken kernelmatrix!(tmp_square, ks, x) ≈ kernelmatrix(ks, x)

            tmp_diag = Vector{Float64}(undef, length(x))
            @test_broken kerneldiagmatrix!(tmp_diag, ks, x) ≈ kerneldiagmatrix(ks, x)
        end
    end
    test_ADs(x->exp(x[1]) * SqExponentialKernel(), rand(1))
end

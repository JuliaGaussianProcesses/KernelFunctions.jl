# Custom Kernel implementation that only defines how to evaluate itself. This is used to
# test that fallback kernelmatrix / kernelmatrix_diag methods work properly.
struct BaseSE <: KernelFunctions.Kernel end
(k::BaseSE)(x, y) = exp(-evaluate(SqEuclidean(), x, y) / 2)

# Custom kernel to test `SimpleKernel` interface on, independently the `SimpleKernel`s that
# are implemented in the package. That this happens to be an exponentiated quadratic kernel
# is a complete coincidence.
struct ToySimpleKernel <: SimpleKernel end
KernelFunctions.metric(::ToySimpleKernel) = SqEuclidean()
KernelFunctions.kappa(::ToySimpleKernel, d) = exp(-d / 2)

@testset "kernelmatrix" begin
    @testset "Kernels" begin
        rng = MersenneTwister(123456)
        k = BaseSE()
        k_se = SqExponentialKernel()

        Nx = 5
        Ny = 3
        D = 2

        vecs = (randn(rng, Nx), randn(rng, Ny))
        colvecs = (ColVecs(randn(rng, D, Nx)), ColVecs(randn(rng, D, Ny)))
        rowvecs = (RowVecs(randn(rng, Nx, D)), RowVecs(randn(rng, Ny, D)))

        @testset "$(typeof(x))" for (x, y) in [vecs, colvecs, rowvecs]
            @test kernelmatrix(k_se, x, y) ≈ kernelmatrix(k, x, y)

            @test kernelmatrix(k, x) ≈ kernelmatrix(k, x, x)

            @test kernelmatrix(k, x, y) ≈ transpose(kernelmatrix(k, y, x))

            @test kernelmatrix_diag(k, x) ≈ diag(kernelmatrix(k, x))

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test kernelmatrix!(tmp, k, x, y) ≈ kernelmatrix(k, x, y)
            @test tmp ≈ kernelmatrix(k, x, y)

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test kernelmatrix!(tmp_square, k, x) ≈ kernelmatrix(k, x)
            @test tmp_square ≈ kernelmatrix(k, x)

            tmp_diag = Vector{Float64}(undef, length(x))
            @test kernelmatrix_diag!(tmp_diag, k, x) ≈ kernelmatrix_diag(k, x)
            @test tmp_diag ≈ kernelmatrix_diag(k, x)
            tmp_diag = Vector{Float64}(undef, length(x))
            @test kernelmatrix_diag!(tmp_diag, k, x, x) ≈ kernelmatrix_diag(k, x, x)
            @test tmp_diag ≈ kernelmatrix_diag(k, x, x)
        end
    end

    @testset "SimpleKernels" begin
        rng = MersenneTwister(123456)
        k = ToySimpleKernel()

        Nx = 5
        Ny = 3
        D = 2

        vecs = (randn(rng, Nx), randn(rng, Ny), ColVecs(randn(rng, D, Ny)))
        colvecs = (
            x=ColVecs(randn(rng, D, Nx)),
            y=ColVecs(randn(rng, D, Ny)),
            x_bad=ColVecs(randn(rng, D + 1, Nx)),
        )
        rowvecs = (
            x=RowVecs(randn(rng, Nx, D)),
            y=RowVecs(randn(rng, Ny, D)),
            x_bad=RowVecs(randn(rng, Ny, D + 1)),
        )

        @testset "$(typeof(x))" for (x, y, x_bad) in [vecs, colvecs, rowvecs]
            @test kernelmatrix(k, x, y) ≈ transpose(kernelmatrix(k, y, x))
            @test_throws DimensionMismatch kernelmatrix(k, x, x_bad)

            @test kernelmatrix(k, x) ≈ kernelmatrix(k, x, x)

            @test kernelmatrix_diag(k, x) ≈ diag(kernelmatrix(k, x))

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test kernelmatrix!(tmp, k, x, y) ≈ kernelmatrix(k, x, y)
            @test tmp ≈ kernelmatrix(k, x, y)
            @test_throws DimensionMismatch kernelmatrix!(tmp, k, x, x)
            @test_throws DimensionMismatch kernelmatrix!(tmp, k, x, x_bad)

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test kernelmatrix!(tmp_square, k, x) ≈ kernelmatrix(k, x)
            @test tmp_square ≈ kernelmatrix(k, x)
            @test_throws DimensionMismatch kernelmatrix!(tmp_square, k, y)

            tmp_diag = Vector{Float64}(undef, length(x))
            @test kernelmatrix_diag!(tmp_diag, k, x) ≈ kernelmatrix_diag(k, x)
            @test tmp_diag ≈ kernelmatrix_diag(k, x)
            @test_throws DimensionMismatch kernelmatrix_diag!(tmp_diag, k, y)
        end
    end

    @testset "AbstractMatrix inputs" begin
        rng = MersenneTwister(123456)
        k = ToySimpleKernel()

        Nx = 4
        Ny = 6
        D = 3

        data = (
            (obsdim=1, x=RowVecs(randn(rng, Nx, D)), y=RowVecs(randn(rng, Ny, D))),
            (obsdim=2, x=ColVecs(randn(rng, D, Nx)), y=ColVecs(randn(rng, D, Ny))),
        )

        @testset "obsdim = $(d.obsdim)" for d in data
            obsdim = d.obsdim
            x = d.x
            X = x.X
            y = d.y
            Y = y.X

            @test kernelmatrix(k, x, y) == kernelmatrix(k, X, Y; obsdim=obsdim)

            @test kernelmatrix(k, x) ≈ kernelmatrix(k, X; obsdim=obsdim)

            @test kernelmatrix_diag(k, x) ≈ kernelmatrix_diag(k, X; obsdim=obsdim)

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test kernelmatrix(k, x, y) ≈ kernelmatrix!(tmp, k, X, Y; obsdim=obsdim)
            @test kernelmatrix(k, x, y) ≈ tmp

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test kernelmatrix(k, x) ≈ kernelmatrix!(tmp_square, k, X; obsdim=obsdim)
            @test kernelmatrix(k, x) ≈ tmp_square

            tmp_diag = Vector{Float64}(undef, length(x))
            @test kernelmatrix_diag(k, x) ≈
                kernelmatrix_diag!(tmp_diag, k, X; obsdim=obsdim)
            @test kernelmatrix_diag(k, x) ≈ tmp_diag
            tmp_diag = Vector{Float64}(undef, length(x))
            @test kernelmatrix_diag!(tmp_diag, k, X, X; obsdim=obsdim) ≈
                kernelmatrix_diag(k, x, x)
            @test tmp_diag ≈ kernelmatrix_diag(k, x, x)
        end

        @testset "deprecated default" begin
            X = randn(rng, D, Nx)
            Y = randn(rng, D, Ny)
            x = ColVecs(X)
            y = ColVecs(Y)

            @test kernelmatrix(k, x, y) == @test_deprecated(kernelmatrix(k, X, Y))

            @test kernelmatrix(k, x) ≈ @test_deprecated(kernelmatrix(k, X))

            @test kernelmatrix_diag(k, x) ≈ @test_deprecated(kernelmatrix_diag(k, X))

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test kernelmatrix(k, x, y) ≈ @test_deprecated(kernelmatrix!(tmp, k, X, Y))
            @test kernelmatrix(k, x, y) ≈ tmp

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test kernelmatrix(k, x) ≈ @test_deprecated(kernelmatrix!(tmp_square, k, X))
            @test kernelmatrix(k, x) ≈ tmp_square

            tmp_diag = Vector{Float64}(undef, length(x))
            @test kernelmatrix_diag(k, x) ≈
                @test_deprecated(kernelmatrix_diag!(tmp_diag, k, X))
            @test kernelmatrix_diag(k, x) ≈ tmp_diag
            tmp_diag = Vector{Float64}(undef, length(x))
            @test @test_deprecated(kernelmatrix_diag!(tmp_diag, k, X, X)) ≈
                kernelmatrix_diag(k, x, x)
            @test tmp_diag ≈ kernelmatrix_diag(k, x, x)
        end
    end

    @testset "Multi Output Kernels" begin
        x = MOInput([rand(5) for _ in 1:4], 3)
        y = MOInput([rand(5) for _ in 1:4], 3)

        k = IndependentMOKernel(GaussianKernel())
        @test kernelmatrix(k, x, y) == k.(collect(x), permutedims(collect(y)))
        @test kernelmatrix(k, x, x) == kernelmatrix(k, x)
    end
end

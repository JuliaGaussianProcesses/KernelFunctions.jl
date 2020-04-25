# Custom kernel to test `SimpleKernel` interface on, independently the `SimpleKernel`s that
# are implemented in the package. That this happens to be an exponentiated quadratic kernel
# is a complete coincidence.
struct ToySimpleKernel <: SimpleKernel end
KernelFunctions.metric(::ToySimpleKernel) = SqEuclidean()
KernelFunctions.kappa(::ToySimpleKernel, d) = exp(-d / 2)

@testset "kernelmatrix" begin

    @testset "SimpleKernels" begin
        rng = MersenneTwister(123456)
        k = ToySimpleKernel()

        Nx = 5
        Ny = 3
        D = 2

        vecs = (randn(rng, Nx), randn(rng, Ny))
        colvecs = (ColVecs(randn(rng, D, Nx)), ColVecs(randn(rng, D, Ny)))
        rowvecs = (RowVecs(randn(rng, Nx, D)), RowVecs(randn(rng, Ny, D)))

        @testset "$(typeof(x))" for (x, y) in [vecs, colvecs, rowvecs]

            @test kernelmatrix(k, x) ≈ kernelmatrix(k, x, x)

            @test kernelmatrix(k, x, y) ≈ transpose(kernelmatrix(k, y, x))

            @test kerneldiagmatrix(k, x) ≈ diag(kernelmatrix(k, x))

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test kernelmatrix!(tmp, k, x, y) ≈ kernelmatrix(k, x, y)

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test kernelmatrix!(tmp_square, k, x) ≈ kernelmatrix(k, x)

            tmp_diag = Vector{Float64}(undef, length(x))
            @test kerneldiagmatrix!(tmp_diag, k, x) ≈ kerneldiagmatrix(k, x)
        end
    end

    @testset "AbstractMatrix inputs" begin
        rng = MersenneTwister(123456)
        k = ToySimpleKernel()

        Nx = 4
        Ny = 6
        D = 3

        data = (
            (obsdim = 1, x = RowVecs(randn(rng, Nx, D)), y = RowVecs(randn(rng, Ny, D))),
            (obsdim = 2, x = ColVecs(randn(rng, D, Nx)), y = ColVecs(randn(rng, D, Ny))),
        )

        @testset "obsdim = $(d.obsdim)" for d in data
            obsdim = d.obsdim
            x = d.x
            X = x.X
            y = d.y
            Y = y.X

            @test kernelmatrix(k, x, y) == kernelmatrix(k, X, Y; obsdim=obsdim)

            @test kernelmatrix(k, x) ≈ kernelmatrix(k, X; obsdim=obsdim)

            @test kerneldiagmatrix(k, x) ≈ kerneldiagmatrix(k, X; obsdim=obsdim)

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test kernelmatrix(k, x, y) ≈ kernelmatrix!(tmp, k, X, Y; obsdim=obsdim)

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test kernelmatrix(k, x) ≈ kernelmatrix!(tmp_square, k, X; obsdim=obsdim)

            tmp_diag = Vector{Float64}(undef, length(x))
            @test kerneldiagmatrix(k, x) ≈ kerneldiagmatrix!(tmp_diag, k, X; obsdim=obsdim)
        end
    end
end

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

    # @testset "Transformed Kernel Matrix Operations" begin
    #     @testset "Inplace Kernel Matrix" begin
    #         for obsdim in [1,2]
    #             @test kernelmatrix!(K[obsdim],kt,A,B,obsdim=obsdim) == kernelmatrix(k,s*A,s*B,obsdim=obsdim)
    #             @test kernelmatrix!(K[obsdim],kt,A,obsdim=obsdim) == kernelmatrix(k,s*A,obsdim=obsdim)
    #             @test kerneldiagmatrix!(Kdiag[obsdim],kt,A,obsdim=obsdim) == kerneldiagmatrix(k,s*A,obsdim=obsdim)
    #         end
    #     end
    #     @testset "Kernel matrix" begin
    #         for obsdim in [1,2]
    #             @test kernelmatrix(kt,A,B,obsdim=obsdim) == kernelmatrix(k,s*A,s*B,obsdim=obsdim)
    #             @test kernelmatrix(kt,A,obsdim=obsdim) == kernelmatrix(k,s*A,obsdim=obsdim)
    #             @test kerneldiagmatrix(kt,A,obsdim=obsdim) == kerneldiagmatrix(k,s*A,obsdim=obsdim)
    #         end
    #     end
    # end
    # @testset "KernelSum" begin
    #     k1 = SqExponentialKernel()
    #     k2 = LinearKernel()
    #     ks = k1 + k2
    #     w1 = 0.4; w2 = 1.2;
    #     ks2 = KernelSum([k1,k2],weights=[w1,w2])
    #     @test all(kernelmatrix(ks,A) .== kernelmatrix(k1,A) + kernelmatrix(k2,A))
    #     @test all(kernelmatrix(ks+k1,A) .≈ 2*kernelmatrix(k1,A) + kernelmatrix(k2,A))
    #     @test all(kernelmatrix(k1+ks,A) .≈ 2*kernelmatrix(k1,A) + kernelmatrix(k2,A))
    #     @test all(kernelmatrix(ks,A,B) .== kernelmatrix(k1,A,B) + kernelmatrix(k2,A,B))
    #     @test all(kerneldiagmatrix(ks,A) .== kerneldiagmatrix(k1,A) + kerneldiagmatrix(k2,A))
    #     @test all(kernelmatrix(ks2,A) .== w1*kernelmatrix(k1,A) + w2*kernelmatrix(k2,A))
    # end
    # @testset "KernelProduct" begin
    #     k1 = SqExponentialKernel()
    #     k2 = LinearKernel()
    #     k3 = RationalQuadraticKernel()
    #     kp = k1 * k2
    #     kp2 = k1 * k3
    #     @test all(kernelmatrix(kp,A) .≈ kernelmatrix(k1,A) .* kernelmatrix(k2,A))
    #     @test all(kernelmatrix(kp*k1,A) .≈ kernelmatrix(k1,A).^2 .* kernelmatrix(k2,A))
    #     @test all(kernelmatrix(k1*kp,A) .≈ kernelmatrix(k1,A).^2 .* kernelmatrix(k2,A))
    #     @test all(kernelmatrix(kp,A) .≈ kernelmatrix(k1,A) .* kernelmatrix(k2,A))
    #     @test all(kernelmatrix(kp,A,B) .≈ kernelmatrix(k1,A,B) .* kernelmatrix(k2,A,B))
    #     @test all(kernelmatrix(kp,A) .≈ kernelmatrix(k1,A) .* kernelmatrix(k2,A))
    #     @test all(kerneldiagmatrix(kp,A) .== kerneldiagmatrix(k1,A) .* kerneldiagmatrix(k2,A))
    # end
end

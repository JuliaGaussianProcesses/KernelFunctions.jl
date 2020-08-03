@testset "kernelproduct" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k3 = RationalQuadraticKernel()
    X = rand(rng, 2,2)

    k = KernelProduct(k1,k2)
    ks1 = 2.0*k1
    ks2 = 0.5*k2
    @test length(k) == 2
    @test k(v1, v2) == (k1 * k2)(v1, v2)
    @test (k * k3)(v1,v2) ≈ (k3 * k)(v1, v2)
    @test (k1 * k2)(v1, v2) == KernelProduct(k1, k2)(v1, v2)
    @test (k * ks1)(v1, v2) ≈ (ks1 * k)(v1, v2)
    @test (k * k)(v1, v2) == KernelProduct([k1, k2, k1, k2])(v1, v2)
    @test KernelProduct([k1, k2]) == KernelProduct((k1, k2)) == k1 * k2

    @test (KernelProduct([k1, k2]) * KernelProduct([k2, k1])).kernels == [k1, k2, k2, k1]
    @test (KernelProduct([k1, k2]) * k3).kernels == [k1, k2, k3]
    @test (k3 * KernelProduct([k1, k2])).kernels == [k3, k1, k2]

    @test (KernelProduct((k1, k2)) * KernelProduct((k2, k1))).kernels == (k1, k2, k2, k1)
    @test (KernelProduct((k1, k2)) * k3).kernels == (k1, k2, k3)
    @test (k3 * KernelProduct((k1, k2))).kernels == (k3, k1, k2)

    @testset "kernelmatrix" begin            
        rng = MersenneTwister(123456)

        Nx = 5
        Ny = 4
        D = 3

        w1 = rand(rng) + 1e-3
        w2 = rand(rng) + 1e-3
        k1 = w1 * SqExponentialKernel()
        k2 = w2 * LinearKernel()
        k = k1 * k2

        @testset "$(typeof(x))" for (x, y) in [
            (ColVecs(randn(rng, D, Nx)), ColVecs(randn(rng, D, Ny))),
            (RowVecs(randn(rng, Nx, D)), RowVecs(randn(rng, Ny, D))),
        ]
            @test kernelmatrix(k, x, y) ≈ kernelmatrix(k1, x, y) .* kernelmatrix(k2, x, y)

            @test kernelmatrix(k, x) ≈ kernelmatrix(k1, x) .* kernelmatrix(k2, x)

            K_diag_manual = kerneldiagmatrix(k1, x) .* kerneldiagmatrix(k2, x)
            @test kerneldiagmatrix(k, x) ≈ K_diag_manual

            tmp = Matrix{Float64}(undef, length(x), length(y))
            @test kernelmatrix!(tmp, k, x, y) ≈ kernelmatrix(k, x, y)

            tmp_square = Matrix{Float64}(undef, length(x), length(x))
            @test kernelmatrix!(tmp_square, k, x) ≈ kernelmatrix(k, x)

            tmp_diag = Vector{Float64}(undef, length(x))
            @test kerneldiagmatrix!(tmp_diag, k, x) ≈ kerneldiagmatrix(k, x)
        end
    end
    test_ADs(x->SqExponentialKernel() * LinearKernel(c= x[1]), rand(1), ADs = [:ForwardDiff, :ReverseDiff])
    @test_broken "Zygote issue"
end

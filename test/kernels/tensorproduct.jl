@testset "tensorproduct" begin
    rng = MersenneTwister(123456)
    u1 = rand(rng, 10)
    u2 = rand(rng, 10)
    v1 = rand(rng, 5)
    v2 = rand(rng, 5)

    # kernels
    k1 = SqExponentialKernel()
    k2 = ExponentialKernel()
    kernel1 = TensorProduct(k1, k2)
    kernel2 = TensorProduct([k1, k2])

    @test kernel1.kernels === (k1, k2) === TensorProduct((k1, k2)).kernels
    @test length(kernel1) == length(kernel2) == 2

    @testset "kappa" begin
        for (x, y) in (((v1, u1), (v2, u2)), ([v1, u1], [v2, u2]))
            val = k1(x[1], y[1]) * k2(x[2], y[2])

            @test kernel1(x, y) == kernel2(x, y) == val
            @test KernelFunctions.kappa(kernel1, x, y) ==
                KernelFunctions.kappa(kernel2, x, y) == val
        end
    end

    @testset "kernelmatrix" begin
        X = rand(rng, 2, 10)
        Y = rand(rng, 2, 10)
        trueX = kernelmatrix(k1, X[1, :]) .* kernelmatrix(k2, X[2, :])
        trueXY = kernelmatrix(k1, X[1, :], Y[1, :]) .* kernelmatrix(k2, X[2, :], Y[2, :])
        tmp = Matrix{Float64}(undef, 10, 10)

        for kernel in (kernel1, kernel2)
            @test kernelmatrix(kernel, X) ≈ trueX
            @test kernelmatrix(kernel, X'; obsdim = 1) ≈ trueX

            @test kernelmatrix(kernel, X, Y) ≈ trueXY
            @test kernelmatrix(kernel, X', Y'; obsdim = 1) ≈ trueXY

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, X)
            @test tmp ≈ trueX

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, X'; obsdim = 1)
            @test tmp ≈ trueX

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, X, Y)
            @test tmp ≈ trueXY

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, X', Y'; obsdim = 1)
            @test tmp ≈ trueXY
        end
    end

    @testset "kerneldiagmatrix" begin
        X = rand(rng, 2, 10)
        trueval = ones(10)
        tmp = Vector{Float64}(undef, 10)

        for kernel in (kernel1, kernel2)
            @test kerneldiagmatrix(kernel, X) == trueval
            @test kerneldiagmatrix(kernel, X'; obsdim = 1) == trueval

            fill!(tmp, 0)
            kerneldiagmatrix!(tmp, kernel, X)
            @test tmp == trueval

            fill!(tmp, 0)
            kerneldiagmatrix!(tmp, kernel, X'; obsdim = 1)
            @test tmp == trueval
        end
    end

    @testset "single kernel" begin
        kernel = TensorProduct(k1)
        @test length(kernel) == 1

        @testset "kappa" begin
            for (x, y) in (((v1,), (v2,)), ([v1], [v2]))
                val = k1(x[1], y[1])

                @test kernel(x, y) == val
                @test KernelFunctions.kappa(kernel, x, y) == val
            end
        end

        @testset "kernelmatrix" begin
            X = rand(rng, 1, 10)
            Y = rand(rng, 1, 10)
            trueX = kernelmatrix(k1, X)
            trueXY = kernelmatrix(k1, X, Y)
            tmp = Matrix{Float64}(undef, 10, 10)

            @test kernelmatrix(kernel, X) ≈ trueX
            @test kernelmatrix(kernel, X'; obsdim = 1) ≈ trueX

            @test kernelmatrix(kernel, X, Y) ≈ trueXY
            @test kernelmatrix(kernel, X', Y'; obsdim = 1) ≈ trueXY

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, X)
            @test tmp ≈ trueX

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, X'; obsdim = 1)
            @test tmp ≈ trueX

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, X, Y)
            @test tmp ≈ trueXY

            fill!(tmp, 0)
            kernelmatrix!(tmp, kernel, X', Y'; obsdim = 1)
            @test tmp ≈ trueXY
        end

        @testset "kerneldiagmatrix" begin
            X = rand(rng, 1, 10)
            trueval = ones(10)
            tmp = Vector{Float64}(undef, 10)

            @test kerneldiagmatrix(kernel, X) == trueval
            @test kerneldiagmatrix(kernel, X'; obsdim = 1) == trueval

            fill!(tmp, 0)
            kerneldiagmatrix!(tmp, kernel, X)
            @test tmp == trueval

            fill!(tmp, 0)
            kerneldiagmatrix!(tmp, kernel, X'; obsdim = 1)
            @test tmp == trueval
        end
    end
end

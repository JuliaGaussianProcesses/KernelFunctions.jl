### Needs to be kept out for julia 1.0

struct baseSE <: KernelFunctions.BaseKernel end
(k::baseSE)(x, y) = exp(-evaluate(SqEuclidean(), x, y))

@testset "kernelmatrix" begin

    rng = MersenneTwister(123456)
    dims = [10,5]
    vA = [rand(rng, dims[1]) for _ in 1:dims[2]]
    A = hcat(vA...)
    vB = [rand(rng, dims[1]) for _ in 1:dims[2]]
    B = hcat(vB...)
    x = rand(rng, dims[1])
    X = collect(reshape(x, 1, :))
    y = rand(rng, dims[2])
    Y = collect(reshape(y, 1 , :))
    KX = zeros(dims[1], dims[1])
    KXY = zeros(dims[1], dims[2])
    C = rand(rng, 8, 9)
    K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
    Kdiag = [zeros(dims[1]),zeros(dims[2])]
    s = rand(rng)
    k = SqExponentialKernel()
    newk = baseSE()
    kt = transform(SqExponentialKernel(),s)

    @testset "Kernel Matrix Operations" begin
        @testset "Inplace Kernel Matrix" begin
            @test kernelmatrix!(KX, k, x) ≈ kernelmatrix!(KX, k, X)
            @test kernelmatrix!(KXY, k, x, y) ≈ kernelmatrix!(KXY, k, X, Y)
            @test kernelmatrix!(K[2], k, vA) ≈ kernelmatrix(k, A) atol = 1e-5
            @test kernelmatrix!(K[2], k, vA, vB) ≈ kernelmatrix(k, A, B) atol = 1e-5
            for obsdim in [1,2]
                @test kernelmatrix!(K[obsdim], k, A, B, obsdim = obsdim) == kernelmatrix(k, A, B, obsdim = obsdim)
                @test kernelmatrix!(K[obsdim], k, A, obsdim = obsdim) == kernelmatrix(k, A, obsdim = obsdim)
                @test kerneldiagmatrix!(Kdiag[obsdim], k, A, obsdim = obsdim) == kerneldiagmatrix(k, A, obsdim = obsdim)
                @test_throws DimensionMismatch kernelmatrix!(K[obsdim], k, A, C, obsdim=obsdim)
                @test_throws DimensionMismatch kernelmatrix!(K[obsdim], k, C, obsdim=obsdim)
                @test_throws DimensionMismatch kerneldiagmatrix!(Kdiag[obsdim], k, C, obsdim=obsdim)
                @test kernelmatrix!(K[obsdim], newk, A, B, obsdim = obsdim) ≈ kernelmatrix(k, A, B, obsdim = obsdim)
                @test kernelmatrix!(K[obsdim], newk, A, obsdim = obsdim) ≈ kernelmatrix(k, A, obsdim = obsdim)
                @test kerneldiagmatrix!(Kdiag[obsdim], newk, A, obsdim = obsdim) ≈ kerneldiagmatrix(k, A, obsdim = obsdim)
            end
        end
        @testset "Kernel matrix" begin
            @test kernelmatrix(k, x) ≈ kernelmatrix(k, X)
            @test kernelmatrix(k, x, y) ≈ kernelmatrix(k, X, Y)
            @test kernelmatrix(k, vA) ≈ kernelmatrix(k, A) atol = 1e-5
            @test kernelmatrix(k, vA, vB) ≈ kernelmatrix(k, A, B) atol = 1e-5
            for obsdim in [1,2]
                @test kerneldiagmatrix(k,A,obsdim=obsdim) == diag(kernelmatrix(k,A,obsdim=obsdim))
                @test k(A,B,obsdim=obsdim) == kernelmatrix(k,A,B,obsdim=obsdim)
                @test k(A,obsdim=obsdim) == kernelmatrix(k,A,obsdim=obsdim)
                @test_throws DimensionMismatch kernelmatrix(k,A,C,obsdim=obsdim)
                @test kernelmatrix(newk, A, B, obsdim = obsdim) ≈ kernelmatrix(k, A, B, obsdim = obsdim)
                @test kernelmatrix(newk, A, obsdim = obsdim) ≈ kernelmatrix(k, A, obsdim = obsdim)
                @test kerneldiagmatrix(newk, A, obsdim = obsdim) ≈ kerneldiagmatrix(k, A, obsdim = obsdim)
            end
        end
        @testset "Transformed Kernel Matrix Operations" begin
            @testset "Inplace Kernel Matrix" begin
                for obsdim in [1,2]
                    @test kernelmatrix!(K[obsdim],kt,A,B,obsdim=obsdim) == kernelmatrix(k,s*A,s*B,obsdim=obsdim)
                    @test kernelmatrix!(K[obsdim],kt,A,obsdim=obsdim) == kernelmatrix(k,s*A,obsdim=obsdim)
                    @test kerneldiagmatrix!(Kdiag[obsdim],kt,A,obsdim=obsdim) == kerneldiagmatrix(k,s*A,obsdim=obsdim)
                end
            end
            @testset "Kernel matrix" begin
                for obsdim in [1,2]
                    @test kernelmatrix(kt,A,B,obsdim=obsdim) == kernelmatrix(k,s*A,s*B,obsdim=obsdim)
                    @test kernelmatrix(kt,A,obsdim=obsdim) == kernelmatrix(k,s*A,obsdim=obsdim)
                    @test kerneldiagmatrix(kt,A,obsdim=obsdim) == kerneldiagmatrix(k,s*A,obsdim=obsdim)
                end
            end
        end
        @testset "KernelSum" begin
            k1 = SqExponentialKernel()
            k2 = LinearKernel()
            ks = k1 + k2
            w1 = 0.4; w2 = 1.2;
            ks2 = KernelSum([k1,k2],weights=[w1,w2])
            @test all(kernelmatrix(ks,A) .== kernelmatrix(k1,A) + kernelmatrix(k2,A))
            @test all(kernelmatrix(ks+k1,A) .≈ 2*kernelmatrix(k1,A) + kernelmatrix(k2,A))
            @test all(kernelmatrix(k1+ks,A) .≈ 2*kernelmatrix(k1,A) + kernelmatrix(k2,A))
            @test all(kernelmatrix(ks,A,B) .== kernelmatrix(k1,A,B) + kernelmatrix(k2,A,B))
            @test all(kerneldiagmatrix(ks,A) .== kerneldiagmatrix(k1,A) + kerneldiagmatrix(k2,A))
            @test all(kernelmatrix(ks2,A) .== w1*kernelmatrix(k1,A) + w2*kernelmatrix(k2,A))
        end
        @testset "KernelProduct" begin
            k1 = SqExponentialKernel()
            k2 = LinearKernel()
            k3 = RationalQuadraticKernel()
            kp = k1 * k2
            kp2 = k1 * k3
            @test all(kernelmatrix(kp,A) .≈ kernelmatrix(k1,A) .* kernelmatrix(k2,A))
            @test all(kernelmatrix(kp*k1,A) .≈ kernelmatrix(k1,A).^2 .* kernelmatrix(k2,A))
            @test all(kernelmatrix(k1*kp,A) .≈ kernelmatrix(k1,A).^2 .* kernelmatrix(k2,A))
            @test all(kernelmatrix(kp,A) .≈ kernelmatrix(k1,A) .* kernelmatrix(k2,A))
            @test all(kernelmatrix(kp,A,B) .≈ kernelmatrix(k1,A,B) .* kernelmatrix(k2,A,B))
            @test all(kernelmatrix(kp,A) .≈ kernelmatrix(k1,A) .* kernelmatrix(k2,A))
            @test all(kerneldiagmatrix(kp,A) .== kerneldiagmatrix(k1,A) .* kerneldiagmatrix(k2,A))
        end
    end
end

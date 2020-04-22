@testset "kernelmatrix" begin

    rng = MersenneTwister(123456)
    dims = [10,5]

    A = rand(rng, dims...)
    B = rand(rng, dims...)
    C = rand(rng, 8, 9)
    K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
    Kdiag = [zeros(dims[1]),zeros(dims[2])]
    s = rand(rng)
    k = SqExponentialKernel()
    kt = transform(SqExponentialKernel(),s)

    @testset "Kernel Matrix Operations" begin
        @testset "Inplace Kernel Matrix" begin
            for obsdim in [1,2]
                @test kernelmatrix!(K[obsdim], k, A, B, obsdim = obsdim) == kernelmatrix(k, A, B, obsdim = obsdim)
                @test kernelmatrix!(K[obsdim], k, A, obsdim = obsdim) == kernelmatrix(k, A, obsdim = obsdim)
                @test kerneldiagmatrix!(Kdiag[obsdim], k, A, obsdim = obsdim) == kerneldiagmatrix(k, A, obsdim = obsdim)
                @test_throws DimensionMismatch kernelmatrix!(K[obsdim], k, A, C, obsdim=obsdim)
                @test_throws DimensionMismatch kernelmatrix!(K[obsdim], k, C, obsdim=obsdim)
                @test_throws DimensionMismatch kerneldiagmatrix!(Kdiag[obsdim], k, C, obsdim=obsdim)
            end
        end
        @testset "Kernel matrix" begin
            for obsdim in [1,2]
                @test kernelmatrix(k,A,B,obsdim=obsdim) == kappa.(k,pairwise(KernelFunctions.metric(k),A,B,dims=obsdim))
                @test kernelmatrix(k,A,obsdim=obsdim) == kappa.(k,pairwise(KernelFunctions.metric(k),A,dims=obsdim))
                @test kerneldiagmatrix(k,A,obsdim=obsdim) == diag(kernelmatrix(k,A,obsdim=obsdim))
                @test k(A,B,obsdim=obsdim) == kernelmatrix(k,A,B,obsdim=obsdim)
                @test k(A,obsdim=obsdim) == kernelmatrix(k,A,obsdim=obsdim)
                # @test KernelFunctions._kernel(k,1.0,2.0) == KernelFunctions._kernel(k,[1.0],[2.0])
                @test_throws DimensionMismatch kernelmatrix(k,A,C,obsdim=obsdim)
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

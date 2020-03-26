using Test
using LinearAlgebra
using KernelFunctions
using SpecialFunctions

x = rand()*2; v1 = rand(3); v2 = rand(3); id = IdentityTransform()
@testset "Kappa functions of kernels" begin
    @testset "Constant" begin
        @testset "ZeroKernel" begin
            k = ZeroKernel()
            @test eltype(k) == Any
            @test kappa(k,2.0) == 0.0
        end
        @testset "WhiteKernel" begin
            k = WhiteKernel()
            @test eltype(k) == Any
            @test kappa(k,1.0) == 1.0
            @test kappa(k,0.0) == 0.0
            @test EyeKernel == WhiteKernel
        end
        @testset "ConstantKernel" begin
            c = 2.0
            k = ConstantKernel(c=c)
            @test eltype(k) == Any
            @test kappa(k,1.0) == c
            @test kappa(k,0.5) == c
        end
    end
    @testset "FBM" begin
        k = FBMKernel(h=0.3)
        @test k(v1,v2) ≈ (sqeuclidean(v1, zero(v1))^0.3 + sqeuclidean(v2, zero(v2))^0.3 - sqeuclidean(v1-v2, zero(v1-v2))^0.3)/2 atol=1e-5
        
        # kernelmatrix tests
        m1 = rand(3,3)
        m2 = rand(3,3)
        @test kernelmatrix(k, m1, m1) ≈ kernelmatrix(k, m1) atol=1e-5
        @test kernelmatrix(k, m1, m2) ≈ k(m1, m2) atol=1e-5

        
        x1 = rand()
        x2 = rand()
        @test kernelmatrix(k, x1*ones(1,1), x2*ones(1,1))[1] ≈ k(x1, x2) atol=1e-5
    end
    @testset "Cosine" begin
        k = CosineKernel()
        @test eltype(k) == Any
        @test kappa(k, 1.0) ≈ -1.0 atol=1e-5
        @test kappa(k, 2.0) ≈ 1.0 atol=1e-5
        @test kappa(k, 1.5) ≈ 0.0 atol=1e-5
        @test kappa(k,x) ≈ cospi(x) atol=1e-5
        @test k(v1, v2) ≈ cospi(sqrt(sum(abs2.(v1-v2)))) atol=1e-5
    end
    @testset "Exponential" begin
        @testset "SqExponentialKernel" begin
            k = SqExponentialKernel()
            @test kappa(k,x) ≈ exp(-x)
            @test k(v1,v2) ≈ exp(-norm(v1-v2)^2)
            @test kappa(SqExponentialKernel(),x) == kappa(k,x)
        end
        @testset "ExponentialKernel" begin
            k = ExponentialKernel()
            @test kappa(k,x) ≈ exp(-x)
            @test k(v1,v2) ≈ exp(-norm(v1-v2))
            @test kappa(ExponentialKernel(),x) == kappa(k,x)
        end
        @testset "GammaExponentialKernel" begin
            γ = 2.0
            k = GammaExponentialKernel(γ=γ)
            @test kappa(k,x) ≈ exp(-(x)^(γ))
            @test k(v1,v2) ≈ exp(-norm(v1-v2)^(2γ))
            @test kappa(GammaExponentialKernel(),x) == kappa(k,x)
            @test GammaExponentialKernel(gamma=γ).γ == [γ]
            #Coherence :
            @test KernelFunctions._kernel(GammaExponentialKernel(γ=1.0),v1,v2) ≈ KernelFunctions._kernel(SqExponentialKernel(),v1,v2)
            @test KernelFunctions._kernel(GammaExponentialKernel(γ=0.5),v1,v2) ≈ KernelFunctions._kernel(ExponentialKernel(),v1,v2)
        end
    end
    @testset "Exponentiated" begin
        @testset "ExponentiatedKernel" begin
            k = ExponentiatedKernel()
            @test kappa(k,x) ≈ exp(x)
            @test kappa(k,-x) ≈ exp(-x)
            @test k(v1,v2) ≈ exp(dot(v1,v2))
        end
    end
    @testset "Matern" begin
        @testset "MaternKernel" begin
            ν = 2.0
            k = MaternKernel(ν=ν)
            matern(x,ν) = 2^(1-ν)/gamma(ν)*(sqrt(2ν)*x)^ν*besselk(ν,sqrt(2ν)*x)
            @test MaternKernel(nu=ν).ν == [ν]
            @test kappa(k,x) ≈ matern(x,ν)
            @test kappa(k,0.0) == 1.0
            @test kappa(MaternKernel(ν=ν),x) == kappa(k,x)
        end
        @testset "Matern32Kernel" begin
            k = Matern32Kernel()
            @test kappa(k,x) ≈ (1+sqrt(3)*x)exp(-sqrt(3)*x)
            @test k(v1,v2) ≈ (1+sqrt(3)*norm(v1-v2))exp(-sqrt(3)*norm(v1-v2))
            @test kappa(Matern32Kernel(),x) == kappa(k,x)
        end
        @testset "Matern52Kernel" begin
            k = Matern52Kernel()
            @test kappa(k,x) ≈ (1+sqrt(5)*x+5/3*x^2)exp(-sqrt(5)*x)
            @test k(v1,v2) ≈ (1+sqrt(5)*norm(v1-v2)+5/3*norm(v1-v2)^2)exp(-sqrt(5)*norm(v1-v2))
            @test kappa(Matern52Kernel(),x) == kappa(k,x)
        end
        @testset "Coherence Materns" begin
            @test kappa(MaternKernel(ν=0.5),x) ≈ kappa(ExponentialKernel(),x)
            @test kappa(MaternKernel(ν=1.5),x) ≈ kappa(Matern32Kernel(),x)
            @test kappa(MaternKernel(ν=2.5),x) ≈ kappa(Matern52Kernel(),x)
        end
    end
    @testset "Polynomial" begin
        c = randn();
        @testset "LinearKernel" begin
            k = LinearKernel()
            @test kappa(k,x) ≈ x
            @test k(v1,v2) ≈ dot(v1,v2)
            @test kappa(LinearKernel(),x) == kappa(k,x)
        end
        @testset "PolynomialKernel" begin
            k = PolynomialKernel()
            @test kappa(k,x) ≈ x^2
            @test k(v1,v2) ≈ dot(v1,v2)^2
            @test kappa(PolynomialKernel(),x) == kappa(k,x)
            #Coherence test
            @test kappa(PolynomialKernel(d=1.0,c=c),x) ≈ kappa(LinearKernel(c=c),x)
        end
    end
    @testset "Mahalanobis" begin
        P = rand(3,3)
        k = MahalanobisKernel(P)
        @test kappa(k,x) == exp(-x)
        @test k(v1,v2) ≈ exp(-sqmahalanobis(v1,v2, k.P))
        @test kappa(ExponentialKernel(),x) == kappa(k,x)
    end
    @testset "RationalQuadratic" begin
        @testset "RationalQuadraticKernel" begin
            α = 2.0
            k = RationalQuadraticKernel(α=α)
            @test RationalQuadraticKernel(alpha=α).α == [α]
            @test kappa(k,x) ≈ (1.0+x/2.0)^-2
            @test k(v1,v2) ≈ (1.0+norm(v1-v2)^2/2.0)^-2
            @test kappa(RationalQuadraticKernel(α=α),x) == kappa(k,x)
        end
        @testset "GammaRationalQuadraticKernel" begin
            k = GammaRationalQuadraticKernel()
            @test kappa(k,x) ≈ (1.0+x^2.0/2.0)^-2
            @test k(v1,v2) ≈ (1.0+norm(v1-v2)^4.0/2.0)^-2
            @test kappa(GammaRationalQuadraticKernel(),x) == kappa(k,x)
            a = 1.0 + rand()
            @test GammaRationalQuadraticKernel(alpha=a).α == [a]
            #Coherence test
            @test kappa(GammaRationalQuadraticKernel(α=a,γ=1.0),x) ≈ kappa(RationalQuadraticKernel(α=a),x)
        end
    end
    @testset "Transformed/Scaled Kernel" begin
        s = rand()
        v = rand(3)
        k = SqExponentialKernel()
        kt = TransformedKernel(k,ScaleTransform(s))
        ktard = TransformedKernel(k,ARDTransform(v))
        ks = ScaledKernel(k,s)
        @test kappa(kt,v1,v2) == kappa(transform(k,ScaleTransform(s)),v1,v2)
        @test kappa(kt,v1,v2) == kappa(transform(k,s),v1,v2)
        @test kappa(kt,v1,v2) ≈ kappa(k,s*v1,s*v2) atol=1e-5
        @test kappa(ktard,v1,v2) ≈ kappa(transform(k,ARDTransform(v)),v1,v2) atol=1e-5
        @test kappa(ktard,v1,v2) == kappa(transform(k,v),v1,v2)
        @test kappa(ktard,v1,v2) == kappa(k,v.*v1,v.*v2)
        @test KernelFunctions.metric(kt) == KernelFunctions.metric(k)
        @test kappa(ks,x) == s*kappa(k,x)
        @test kappa(ks,x) == kappa(s*k,x)
    end
    @testset "KernelCombinations" begin
        k1 = LinearKernel()
        k2 = SqExponentialKernel()
        k3 = RationalQuadraticKernel()
        X = rand(2,2)
        @testset "KernelSum" begin
            w = [2.0,0.5]
            k = KernelSum([k1,k2],w)
            ks1 = 2.0*k1
            ks2 = 0.5*k2
            @test length(k) == 2
            @test kappa(k,v1,v2) == kappa(2.0*k1+0.5*k2,v1,v2)
            @test kappa(k+k3,v1,v2) ≈ kappa(k3+k,v1,v2)
            @test kappa(k1+k2,v1,v2) == kappa(KernelSum([k1,k2]),v1,v2)
            @test kappa(k+ks1,v1,v2) ≈ kappa(ks1+k,v1,v2)
            @test kappa(k+k,v1,v2) == kappa(KernelSum([k1,k2,k1,k2],vcat(w,w)),v1,v2)
        end
        @testset "KernelProduct" begin
            k = KernelProduct([k1,k2])
            @test length(k) == 2
            @test kappa(k,v1,v2) == kappa(k1*k2,v1,v2)
            @test kappa(k*k,v1,v2) ≈ kappa(k,v1,v2)^2
            @test kappa(k*k3,v1,v2) ≈ kappa(k3*k,v1,v2)
        end
    end
end

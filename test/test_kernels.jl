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
        end
        @testset "ConstantKernel" begin
            c = 2.0
            k = ConstantKernel(c)
            k₂ = ConstantKernel(IdentityTransform(),c)
            @test eltype(k) == Any
            @test kappa(k,1.5)== kappa(k₂,1.5)
            @test kappa(k,1.0) == c
            @test kappa(k,0.5) == c
        end
    end
    @testset "Exponential" begin
        @testset "SqExponentialKernel" begin
            k = SqExponentialKernel()
            @test kappa(k,x) ≈ exp(-x)
            @test k(v1,v2) ≈ exp(-norm(v1-v2)^2)
            @test kappa(SqExponentialKernel(id),x) == kappa(k,x)
            l = 0.5
            k = SqExponentialKernel(l)
            @test k(v1,v2) ≈ exp(-l^2*norm(v1-v2)^2)
            v = rand(3)
            k = SqExponentialKernel(v)
            @test k(v1,v2) ≈ exp(-norm(v.*(v1-v2))^2)
        end
        @testset "ExponentialKernel" begin
            k = ExponentialKernel()
            @test kappa(k,x) ≈ exp(-x)
            @test k(v1,v2) ≈ exp(-norm(v1-v2))
            @test kappa(ExponentialKernel(id),x) == kappa(k,x)
            l = 0.5
            k = ExponentialKernel(l)
            @test k(v1,v2) ≈ exp(-l*norm(v1-v2))
            v = rand(3)
            k = ExponentialKernel(v)
            @test k(v1,v2) ≈ exp(-norm(v.*(v1-v2)))
        end
        @testset "GammaExponentialKernel" begin
            k = GammaExponentialKernel(1.0,2.0)
            @test kappa(k,x) ≈ exp(-(x)^(k.γ))
            @test k(v1,v2) ≈ exp(-norm(v1-v2)^(2k.γ))
            @test kappa(GammaExponentialKernel(id),x) == kappa(k,x)
            l = 0.5
            k = GammaExponentialKernel(l,1.5)
            @test k(v1,v2) ≈ exp(-l^(3.0)*norm(v1-v2)^(3.0))
            v = rand(3)
            k = GammaExponentialKernel(v,3.0)
            @test k(v1,v2) ≈ exp(-norm(v.*(v1-v2)).^6.0)
            #Coherence :
            @test kernel(GammaExponentialKernel(1.0,1.0),v1,v2) ≈ kernel(SqExponentialKernel(),v1,v2)
            @test kernel(GammaExponentialKernel(1.0,0.5),v1,v2) ≈ kernel(ExponentialKernel(),v1,v2)
        end
    end
    @testset "Exponentiated" begin
        @testset "ExponentiatedKernel" begin
            k = ExponentiatedKernel()
            @test kappa(k,x) ≈ exp(x)
            @test kappa(k,-x) ≈ exp(-x)
            @test k(v1,v2) ≈ exp(dot(v1,v2))
            l = 0.5
            k = ExponentiatedKernel(l)
            @test k(v1,v2) ≈ exp(l^2*dot(v1,v2))
            v = rand(3)
            k = ExponentiatedKernel(v)
            @test k(v1,v2) ≈ exp(dot(v.*v1,v.*v2))
        end
    end
    @testset "Matern" begin
        @testset "MaternKernel" begin
            ν = 2.0
            k = MaternKernel(1.0,ν)
            matern(x,ν) = 2^(1-ν)/gamma(ν)*(sqrt(2ν)*x)^ν*besselk(ν,sqrt(2ν)*x)
            @test kappa(k,x) ≈ matern(x,ν)
            @test kappa(k,0.0) == 1.0
            @test kappa(MaternKernel(id,ν),x) == kappa(k,x)
            l = 0.5; ν = 3.0
            k = MaternKernel(l,ν)
            @test k(v1,v2) ≈ matern(l*norm(v1-v2),ν)
            v = rand(3); ν = 2.1
            k = MaternKernel(v,ν)
            @test k(v1,v2) ≈ matern(norm(v.*(v1-v2)),ν)
        end
        @testset "Matern32Kernel" begin
            k = Matern32Kernel()
            @test kappa(k,x) ≈ (1+sqrt(3)*x)exp(-sqrt(3)*x)
            @test k(v1,v2) ≈ (1+sqrt(3)*norm(v1-v2))exp(-sqrt(3)*norm(v1-v2))
            @test kappa(Matern32Kernel(id),x) == kappa(k,x)
            l = 0.5
            k = Matern32Kernel(l)
            @test k(v1,v2) ≈ (1+l*sqrt(3)*norm(v1-v2))exp(-l*sqrt(3)*norm(v1-v2))
            v = rand(3)
            k = Matern32Kernel(v)
            @test k(v1,v2) ≈ (1+sqrt(3)*norm(v.*(v1-v2)))exp(-sqrt(3)*norm(v.*(v1-v2)))
        end
        @testset "Matern52Kernel" begin
            k = Matern52Kernel()
            @test kappa(k,x) ≈ (1+sqrt(5)*x+5/3*x^2)exp(-sqrt(5)*x)
            @test k(v1,v2) ≈ (1+sqrt(5)*norm(v1-v2)+5/3*norm(v1-v2)^2)exp(-sqrt(5)*norm(v1-v2))
            @test kappa(Matern52Kernel(id),x) == kappa(k,x)
            l = 0.5
            k = Matern52Kernel(l)
            @test k(v1,v2) ≈ (1+l*sqrt(5)*norm(v1-v2)+l^2*5/3*norm(v1-v2)^2)exp(-l*sqrt(5)*norm(v1-v2))
            v = rand(3)
            k = Matern52Kernel(v)
            @test k(v1,v2) ≈ (1+sqrt(5)*norm(v.*(v1-v2))+5/3*norm(v.*(v1-v2))^2)exp(-sqrt(5)*norm(v.*(v1-v2)))
        end
        @testset "Coherence Materns" begin
            @test kappa(MaternKernel(1.0,0.5),x) ≈ kappa(ExponentialKernel(),x)
            @test kappa(MaternKernel(1.0,1.5),x) ≈ kappa(Matern32Kernel(),x)
            @test kappa(MaternKernel(1.0,2.5),x) ≈ kappa(Matern52Kernel(),x)
        end
    end
    @testset "Polynomial" begin
        c = randn();
        @testset "LinearKernel" begin
            k = LinearKernel()
            @test kappa(k,x) ≈ x
            @test k(v1,v2) ≈ dot(v1,v2)
            @test kappa(LinearKernel(id),x) == kappa(k,x)
            l = 0.5
            k = LinearKernel(l,c)
            @test k(v1,v2) ≈ l^2*dot(v1,v2) + c
            v = rand(3)
            k = LinearKernel(v,c)
            @test k(v1,v2) ≈ dot(v.*v1,v.*v2) + c
        end
        @testset "PolynomialKernel" begin
            k = PolynomialKernel()
            @test kappa(k,x) ≈ x^2
            @test k(v1,v2) ≈ dot(v1,v2)^2
            @test kappa(PolynomialKernel(id),x) == kappa(k,x)
            d = 3.0
            l = 0.5
            k = PolynomialKernel(l,d,c)
            @test k(v1,v2) ≈ (l^2*dot(v1,v2) + c)^d
            v = rand(3)
            k = PolynomialKernel(v,d,c)
            @test k(v1,v2) ≈ (dot(v.*v1,v.*v2) + c)^d
            #Coherence test
            @test kappa(PolynomialKernel(1.0,1.0,c),x) ≈ kappa(LinearKernel(1.0,c),x)
        end
    end
    @testset "RationalQuadratic" begin
        @testset "RationalQuadraticKernel" begin
            k = RationalQuadraticKernel()
            @test kappa(k,x) ≈ (1.0+x/2.0)^-2
            @test k(v1,v2) ≈ (1.0+norm(v1-v2)^2/2.0)^-2
            @test kappa(RationalQuadraticKernel(id),x) == kappa(k,x)
            l = 0.5
            a = 1.0 + rand()
            k = RationalQuadraticKernel(l,a)
            @test k(v1,v2) ≈ (1.0+l^2*norm(v1-v2)^2/a)^-a
            v = rand(3)
            k = RationalQuadraticKernel(v,a)
            @test k(v1,v2) ≈ (1.0+norm(v.*(v1-v2))^2/a)^-a
        end
        @testset "GammaRationalQuadraticKernel" begin
            k = GammaRationalQuadraticKernel()
            @test kappa(k,x) ≈ (1.0+x^2.0/2.0)^-2
            @test k(v1,v2) ≈ (1.0+norm(v1-v2)^4.0/2.0)^-2
            @test kappa(GammaRationalQuadraticKernel(id),x) == kappa(k,x)
            l = 0.5
            a = 1.0 + rand()
            g = 4.0
            k = GammaRationalQuadraticKernel(l,a,g)
            @test k(v1,v2) ≈ (1.0+(l^2g)*norm(v1-v2)^(2g)/a)^-a
            v = rand(3)
            k = GammaRationalQuadraticKernel(v,a,g)
            @test k(v1,v2) ≈ (1.0+(norm(v.*(v1-v2))^(2g))/a)^-a
            #Coherence test
            @test kappa(GammaRationalQuadraticKernel(1.0,a,1.0),x) ≈ kappa(RationalQuadraticKernel(1.0,a),x)
        end
    end
end

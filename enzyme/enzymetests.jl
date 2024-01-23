using Enzyme
using Zygote
using KernelFunctions

using BenchmarkTools, Test

function enzyme_gradient(f, x)
    dx = zero(x)
    autodiff(Reverse, f, Active, Duplicated(x, dx))
    return dx
end

function zygote_gradient(f, x)
    return only(Zygote.gradient(f, x))
end

function test_and_benchmark(kernel, n = 1000)
    @info "Testing $kernel"
    x = rand(n)
    f(x) = sum(kernelmatrix(kernel, x))
    val = try
        f(x)
    catch
        @error "Error while evaluating function"
        return nothing
    end
    EG = try
        enzyme_gradient(f, x)
    catch
        @error "Error while evaluating gradient with Enzyme"
        return nothing
    end
    ZG = try
        zygote_gradient(f, x)
    catch
        @error "Error while evaluating gradient with Zygote"
        return nothing
    end
    if isnothing(ZG)
        @test iszero(EG)
    else
        @test EG ≈ ZG
    end
    print("        Function evaluation time: "); @btime $f($x);
    print("        Enzyme gradient time:     "); @btime enzyme_gradient($f, $x);
    print("        Zygote gradient time:     "); @btime zygote_gradient($f, $x);
    return nothing
end

@testset "SimpleKernel" begin
    @testset "$(nameof(typeof(kernel)))" for kernel in [
        ConstantKernel(),
        CosineKernel(),
        ExponentialKernel(),
        ExponentiatedKernel(),
        EyeKernel(),
        FBMKernel(),
        GammaExponentialKernel(),
        GammaRationalKernel(),
        GaussianKernel(),
        LaplacianKernel(),
        LinearKernel(),
        Matern12Kernel(),
        Matern32Kernel(),
        Matern52Kernel(),
        MaternKernel(; nu = 2.1),
        NeuralNetworkKernel(),
        PeriodicKernel(),
        PiecewisePolynomialKernel(; dim = 1),
        PolynomialKernel(),
        RBFKernel(),
        RationalKernel(),
        RationalQuadraticKernel(),
        SEKernel(),
        WhiteKernel(),
        WienerKernel(),
        ZeroKernel(),
    ]
        test_and_benchmark(kernel, 1000)
    end
end

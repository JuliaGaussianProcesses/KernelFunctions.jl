using BenchmarkTools
using KernelFunctions
using Zygote
using ForwardDiff

N1 = 10
N2 = 20

X = rand(N1, N2)
Xc = ColVecs(X)
Xr = RowVecs(X)
Xv = collect.(eachcol(X))
Y = rand(N1, N2)
Yc = ColVecs(Y)
Yr = RowVecs(Y)
Yv = collect.(eachcol(Y))

# Create the general suite of benchmarks
SUITE = BenchmarkGroup()

kernels = Dict(
    # Constant Kernels
    "Constant" => ((2.0,), x->ConstantKernel(x)),
    "White" => ((), ()->WhiteKernel()),
    # Cosine Kernel
    "Cosine" => ((), ()->CosineKernel()),
    # Exponential Kernels
    "Exponential" => ((), ()->ExponentialKernel()),
    "Gibbs" => ((), ()->GibbsKernel(;lengthscale=sin)),
    "SqExponential" => ((), ()->SqExponentialKernel()),
    "GammaExponential" => ((1.0,), x->GammaExponentialKernel(;γ=2 * logistic(x))),
    # Exponentiated Kernel
    "Exponentiated" => ((), ()->ExponentiatedKernel()),
)

inputtypes = Dict("ColVecs" => (Xc, Yc), "RowVecs" => (Xr, Yr), "Vecs" => (Xv, Yv))

functions = Dict(
    "kernelmatrixX" => (fk, args, X, Y) -> kernelmatrix(fk(args...), X),
    "kernelmatrixXY" => (fk, args, X, Y) -> kernelmatrix(fk(args...), X, Y),
    "kernelmatrix_diagX" => (fk, args, X, Y) -> kernelmatrix_diag(fk(args...), X),
    "kernelmatrix_diagXY" => (fk, args, X, Y) -> kernelmatrix_diag(fk(args...), X, Y),
)

# Test the allocated functions
SUITE["alloc_suite"] = s_alloc = BenchmarkGroup()
for (kname, (kargs, kf)) in kernels
    s_alloc[kname] = sk = BenchmarkGroup()
    for (inputname, (X, Y)) in inputtypes
        sk[inputname] = si = BenchmarkGroup()
        for (fname, f) in functions
            si[fname] = @benchmarkable $f($kf, $kargs, $X, $Y)
        end
    end
end

# Test the AD frameworks
## Zygote
SUITE["zygote"] = s_zygote = BenchmarkGroup()
for (kname, (kargs, kf)) in kernels
    s_zygote[kname] = sk = BenchmarkGroup()
    for (inputname, (X, Y)) in inputtypes
        sk[inputname] = si = BenchmarkGroup()
        for (fname, f) in functions
            si[fname] = @benchmarkable Zygote.pullback($kargs, $X, $Y) do args, x, Y
                $f($kf(args...), x, y)
            end
        end
    end
end

# Uncomment the following to run benchmark locally

# tune!(SUITE)

# results = run(SUITE, verbose=true)

using BenchmarkTools
using KernelFunctions
using LogExpFunctions: logistic
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

# Create a list of kernel and their constructors
kernels = Dict(
    # Constant Kernels
    "Constant" => ((2.0,), x -> ConstantKernel(; c=x)),
    "White" => ((), () -> WhiteKernel()),
    # Cosine Kernel
    "Cosine" => ((), () -> CosineKernel()),
    # Exponential Kernels
    "Exponential" => ((), () -> ExponentialKernel()),
    "Gibbs" => ((), () -> GibbsKernel(; lengthscale=x -> sin.(x))),
    "SqExponential" => ((), () -> SqExponentialKernel()),
    "GammaExponential" => ((1.0,), x -> GammaExponentialKernel(; γ=2 * logistic(x))),
    # Exponentiated Kernel
    "Exponentiated" => ((), () -> ExponentiatedKernel()),
)

inputtypes = Dict("ColVecs" => (Xc, Yc), "RowVecs" => (Xr, Yr), "Vecs" => (Xv, Yv))

functions = Dict(
    "kernelmatrixX" => (fk, args, X, Y) -> kernelmatrix(fk(args...), X),
    "kernelmatrixXY" => (fk, args, X, Y) -> kernelmatrix(fk(args...), X, Y),
    "kernelmatrix_diagX" => (fk, args, X, Y) -> kernelmatrix_diag(fk(args...), X),
    "kernelmatrix_diagXY" => (fk, args, X, Y) -> kernelmatrix_diag(fk(args...), X, Y),
)

# Test the allocated functions
SUITE["Allocated Functions"] = suite_alloc = BenchmarkGroup()
for (kname, (kargs, kf)) in kernels
    suite_alloc[kname] = suite_kernel = BenchmarkGroup()
    for (inputname, (X, Y)) in inputtypes
        suite_kernel[inputname] = suite_input = BenchmarkGroup()
        for (fname, f) in functions
            suite_input[fname] = @benchmarkable $f($kf, $kargs, $X, $Y)
        end
    end
end

# Test the AD frameworks
## Zygote
SUITE["Zygote"] = suite_zygote = BenchmarkGroup()
for (kname, (kargs, kf)) in kernels
    suite_zygote[kname] = suite_kernel = BenchmarkGroup()
    for (inputname, (X, Y)) in inputtypes
        suite_kernel[inputname] = suite_input = BenchmarkGroup()
        for (fname, f) in functions
            # Forward-pass
            suite_input[fname * "_forward"] = @benchmarkable Zygote.pullback(
                $kargs, $X, $Y
            ) do args, x, y
                $f($kf, args, x, y)
            end
            # Reverse pass
            out, pb = Zygote.pullback(kargs, X, Y) do args, x, y
                f(kf, args, x, y)
            end
            suite_input[fname * "_reverse"] = @benchmarkable $pb($out)
        end
    end
end

## ForwardDiff
# Right now there is no canonical way to turn (kargs, X, Y) into an array.
# SUITE["ForwardDiff"] = suite_forwarddiff = BenchmarkGroup()
# for (kname, (kargs, kf)) in kernels
#     suite_forwarddiff[kname] = suite_kernel = BenchmarkGroup()
#     for (inputname, (X, Y)) in inputtypes
#         suite_kernel[inputname] = suite_input = BenchmarkGroup()
#         for (fname, f) in functions
#             suite_input[fname] = @benchmarkable ForwardDiff.gradient($kargs, $X, $Y) do args, x, y
#                 $f($kf, args, x, y)
#             end
#         end
#     end
# end

# Uncomment the following to run benchmark locally

# tune!(SUITE)

# results = run(SUITE; verbose=true)

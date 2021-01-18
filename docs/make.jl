using Documenter
using Literate
using KernelFunctions

const EXAMPLES_SRC = joinpath(@__DIR__, "..", "examples")
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")

if ispath(EXAMPLES_OUT)
    rm(EXAMPLES_OUT; recursive=true)
end

for filename in readdir(EXAMPLES_SRC)
    endswith(filename, ".jl") || continue
    name = splitext(filename)[1]
    Literate.markdown(
        joinpath(EXAMPLES_SRC, filename), EXAMPLES_OUT; name=name, documenter=true
    )
end

DocMeta.setdocmeta!(
    KernelFunctions,
    :DocTestSetup,
    :(using KernelFunctions, LinearAlgebra, Random);
    recursive=true,
)

makedocs(;
    sitename="KernelFunctions",
    format=Documenter.HTML(),
    modules=[KernelFunctions],
    pages=[
        "Home" => "index.md",
        "User Guide" => "userguide.md",
        "Examples" => [
            "Kernel Ridge Regression" => "examples/kernel_ridge_regression.md",
            "Training kernel parameters" => "examples/train_kernel_parameters.md",
            "Gaussian process priors" => "examples/gaussianprocesspriors.md",
            "SVM" => "examples/svm.md",
            "Deep Kernel Learning" => "examples/deepkernellearning.md",
        ],
        "Kernel Functions" => "kernels.md",
        "Input Transforms" => "transform.md",
        "Metrics" => "metrics.md",
        "Theory" => "theory.md",
        "Custom Kernels" => "create_kernel.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/KernelFunctions.jl.git", push_preview=true
)

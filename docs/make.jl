using Documenter
using KernelFunctions

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
        "Examples" => "example.md",
        "Kernel Functions" => "kernels.md",
        "Input Transforms" => "transform.md",
        "Metrics" => "metrics.md",
        "Theory" => "theory.md",
        "Custom Kernels" => "create_kernel.md",
        "API" => "api.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/KernelFunctions.jl.git", push_preview=true
)

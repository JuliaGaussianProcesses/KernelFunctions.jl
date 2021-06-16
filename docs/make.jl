using Documenter
using Literate

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using KernelFunctions

const EXAMPLES_SRC = joinpath(@__DIR__, "..", "examples")
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
const BLACKLIST = ["deepkernellearning", "kernelridgeregression", "svm"]

if ispath(EXAMPLES_OUT)
    rm(EXAMPLES_OUT; recursive=true)
end

for filepath in readdir(EXAMPLES_SRC; join=true)
    endswith(filepath, ".jl") || continue
    any([occursin(blacklistname, filepath) for blacklistname in BLACKLIST]) && continue
    Literate.markdown(filepath, EXAMPLES_OUT; documenter=true)
    Literate.notebook(filepath, EXAMPLES_OUT; documenter=true)
end

DocMeta.setdocmeta!(
    KernelFunctions,
    :DocTestSetup,
    quote
        using KernelFunctions
        using LinearAlgebra
        using Random
        using PDMats: PDMats
    end;
    recursive=true,
)

makedocs(;
    modules=[KernelFunctions],
    sitename="KernelFunctions.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "userguide.md",
        "kernels.md",
        "transform.md",
        "metrics.md",
        "theory.md",
        "create_kernel.md",
        "API" => "api.md",
        "Kernel Functions" => "kernels.md",
        "Input Transforms" => "transform.md",
        "Metrics" => "metrics.md",
        "Theory" => "theory.md",
        "Custom Kernels" => "create_kernel.md",
        # "Examples" =>
        #     joinpath.(
        #         "examples", filter(filename -> endswith(filename, ".md"), readdir(EXAMPLES_OUT))
        #     ),
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/KernelFunctions.jl.git", push_preview=true
)

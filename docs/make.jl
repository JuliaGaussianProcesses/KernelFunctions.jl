using Documenter

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using KernelFunctions

if ispath(joinpath(@__DIR__, "src", "examples"))
    rm(joinpath(@__DIR__, "src", "examples"), recursive=true)
end

for filename in readdir(joinpath(@__DIR__, "..", "examples"))
    endswith(filename, ".jl") || continue
    name = splitext(filename)[1]
    Literate.markdown(
        joinpath(@__DIR__, "..", "examples", filename),
        joinpath(@__DIR__, "src", "examples"),
        name = name,
        documenter = true,
    )
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
    sitename="KernelFunctions",
    format=Documenter.HTML(),
    modules=[KernelFunctions],
    pages=[
        "Home" => "index.md",
        "userguide.md",
        "kernels.md",
        "transform.md",
        "metrics.md",
        "theory.md",
        "create_kernel.md",
        "API" => "api.md",
        "Examples" => "example.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/KernelFunctions.jl.git", push_preview=true
)

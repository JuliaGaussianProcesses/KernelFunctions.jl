### Process examples
using Pkg
Pkg.add(Pkg.PackageSpec(; url="https://github.com/JuliaGaussianProcesses/JuliaGPsDocs.jl")) # While the package is unregistered, it's a workaround

using JuliaGPsDocs

using KernelFunctions
using PDMats, Kronecker  # we have to load all optional packages to generate the full API documentation

JuliaGPsDocs.generate_examples(KernelFunctions)

### Build documentation
using Documenter

# Doctest setup
DocMeta.setdocmeta!(
    KernelFunctions,
    :DocTestSetup,
    quote
        using KernelFunctions
    end;  # we have to load all packages used (implicitly) within jldoctest blocks in the API docstrings
    recursive=true,
)

makedocs(;
    sitename="KernelFunctions.jl",
    format=Documenter.HTML(),
    modules=[KernelFunctions],
    pages=[
        "Home" => "index.md",
        "userguide.md",
        "kernels.md",
        "transform.md",
        "metrics.md",
        "create_kernel.md",
        "API" => "api.md",
        "Design" => "design.md",
        "Examples" => JuliaGPsDocs.find_generated_examples(KernelFunctions),
    ],
    strict=true,
    checkdocs=:exports,
    doctestfilters=JuliaGPsDocs.DOCTEST_FILTERS,
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/KernelFunctions.jl.git", push_preview=true
)

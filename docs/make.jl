### Process examples
const EXAMPLES_SRC = joinpath(@__DIR__, "..", "examples")
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
const LITERATEJL = joinpath(@__DIR__, "literate.jl")

# Always rerun examples
ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)

# Run examples asynchronously
processes = map(filter!(isdir, readdir(EXAMPLES_SRC; join=true))) do example
    scriptjl = joinpath(example, "script.jl")
    return run(
        pipeline(
            `$(Base.julia_cmd()) $LITERATEJL $scriptjl $EXAMPLES_OUT`;
            stdin=devnull,
            stdout=devnull,
            stderr=stderr,
        );
        wait=false,
    )::Base.Process
end

# Check that all examples were run successfully
isempty(processes) || success(processes) || error("some examples were not run successfully")

### Build documentation
using Documenter

using KernelFunctions
using PDMats, Kronecker  # we have to load all optional packages to generate the full API documentation

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

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
        "Examples" =>
            map(filter!(filename -> endswith(filename, ".md"), readdir(EXAMPLES_OUT))) do x
                return joinpath("examples", x)
            end,
    ],
    strict=true,
    checkdocs=:exports,
    doctestfilters=[
        r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
        r"(Array{[a-zA-Z0-9]+,\s?1}|Vector{[a-zA-Z0-9]+})",
        r"(Array{[a-zA-Z0-9]+,\s?2}|Matrix{[a-zA-Z0-9]+})",
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/KernelFunctions.jl.git", push_preview=true
)

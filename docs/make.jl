using Documenter
using Literate
using Pkg

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

using KernelFunctions

const PACKAGE_DIR = joinpath(@__DIR__, "..")
const EXAMPLES_SRC = joinpath(PACKAGE_DIR, "examples")
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
const BLACKLIST = ["deep-kernel-learning", "support-vector-machine"]

ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)

cmd = Base.julia_cmd()
docs_env = @__DIR__
for example in readdir(EXAMPLES_SRC)
    example ∈ BLACKLIST && continue
    exampledir = joinpath(EXAMPLES_SRC, example)
    isdir(exampledir) || continue
    filepath = joinpath(exampledir, "script.jl")
    @show load_path = "$(exampledir):$(docs_env)"
    code = """
        import Pkg
        Pkg.instantiate()
        using Literate
        Literate.markdown("$(filepath)", "$(EXAMPLES_OUT)"; name="$(example)", documenter=true, execute=true)
        Literate.notebook("$(filepath)", "$(EXAMPLES_OUT)"; name="$(example)", documenter=true, execute=true)
    """
    withenv(JULIA_LOAD_PATH" => load_path) do
        run(`$(cmd) --project=$(exampledir) -e $(code)`)
    end
end
#    Pkg.activate(exampledir) do
#        Pkg.develop(; path=PACKAGE_DIR)
#        Pkg.instantiate()
#        filepath = joinpath(exampledir, "script.jl")
#        Literate.markdown(
#            filepath, EXAMPLES_OUT; name=example, documenter=true, preprocess=preprocess
#        )

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
        "Design" => "design.md",
        "Examples" =>
            joinpath.(
                "examples",
                filter(filename -> endswith(filename, ".md"), readdir(EXAMPLES_OUT)),
            ),
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

using Documenter
using KernelFunctions

makedocs(
    sitename = "KernelFunctions",
    format = Documenter.HTML(),
    modules = [KernelFunctions],
    pages = ["Home"=>"index.md",
             "User Guide" => "userguide.md",
             "Examples"=>"example.md",
             "Kernel Functions"=>"kernels.md",
             "Transform"=>"transform.md",
             "Metrics"=>"metrics.md",
             "Theory"=>"theory.md",
             "Custom Kernels"=>"create_kernel.md"
             "API"=>"api.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/JuliaGaussianProcesses/KernelFunctions.jl.git",
    target = "build"
)

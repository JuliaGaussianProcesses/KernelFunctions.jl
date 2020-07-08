using Documenter
using Literate
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

makedocs(
    sitename = "KernelFunctions",
    format = Documenter.HTML(),
    modules = [KernelFunctions],
    pages = ["Home"=>"index.md",
             "User Guide" => "userguide.md",
             "Examples"=>
                    ["SVM" => "examples/svm.md",
                     # "Kernel Ridge Regression" => "examples/kernelridgeregression.md",
                     # "Deep Kernel Learning" => "examples/deepkernellearning.md",
                     ],
             "Kernel Functions"=>"kernels.md",
             "Transform"=>"transform.md",
             "Metrics"=>"metrics.md",
             "Theory"=>"theory.md",
             "Custom Kernels"=>"create_kernel.md",
             "API"=>"api.md"]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    deps = Deps.pip("mkdocs", "python-markdown-math"),
    repo = "github.com/JuliaGaussianProcesses/KernelFunctions.jl.git",
    target = "build",
    push_preview = true,
)

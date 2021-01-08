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

DocMeta.setdocmeta!(
    KernelFunctions,
    :DocTestSetup,
    :(using KernelFunctions, LinearAlgebra, Random);
    recursive=true,
)

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

deploydocs(;
    repo = "github.com/JuliaGaussianProcesses/KernelFunctions.jl.git",
    push_preview = true,
)

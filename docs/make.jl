using Documenter
using KernelFunctions

makedocs(
    sitename = "KernelFunctions",
    format = Documenter.HTML(),
    modules = [KernelFunctions]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

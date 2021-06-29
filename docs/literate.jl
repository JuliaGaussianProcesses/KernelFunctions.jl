# Retrieve filename of literate script
if length(ARGS) != 2
    error("please specify the literate script and the output directory")
end
const SCRIPTJL = ARGS[1]
const OUTDIR = ARGS[2]

# Activate environment
using Pkg: Pkg
Pkg.activate(dirname(SCRIPTJL))
Pkg.instantiate()
using Literate: Literate

# Add link to nbviewer below the first heading of level 1
function preprocess(content)
    sub = SubstitutionString(
        """
#md # ```@meta
#md # EditURL = "@__REPO_ROOT_URL__/examples/@__NAME__/script.jl"
#md # ```
#md #
\\0
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb)
#md #
# You are seeing the
#md # HTML output generated by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and
#nb # notebook output generated by
# [Literate.jl](https://github.com/fredrikekre/Literate.jl) from the
# [Julia source file](@__REPO_ROOT_URL__/examples/@__NAME__/script.jl).
# The corresponding
#md # notebook can be viewed in [nbviewer](@__NBVIEWER_ROOT_URL__/examples/@__NAME__.ipynb).
#nb # HTML output can be viewed [here](https://juliagaussianprocesses.github.io/KernelFunctions.jl/dev/examples/@__NAME__/).
#
        """,
    )
    return replace(content, r"^# # [^\n]*"m => sub; count=1)
end

# Convert to markdown and notebook
Literate.markdown(
    SCRIPTJL,
    OUTDIR;
    name=basename(dirname(SCRIPTJL)),
    documenter=false,
    execute=true,
    preprocess=preprocess,
)
Literate.notebook(
    SCRIPTJL,
    OUTDIR;
    name=basename(dirname(SCRIPTJL)),
    documenter=false,
    execute=true,
    preprocess=preprocess,
)

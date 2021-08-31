using KernelFunctions
using AxisArrays
using Distances
using Documenter
using Functors: functor
using Kronecker: Kronecker
using LinearAlgebra
using LogExpFunctions
using PDMats
using Random
using SpecialFunctions
using Test
using Zygote: Zygote
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using FiniteDifferences: FiniteDifferences

using KernelFunctions: SimpleKernel, metric, kappa, ColVecs, RowVecs, TestUtils

using KernelFunctions.TestUtils: test_interface

# The GROUP is used to run different sets of tests in parallel on the GitHub Actions CI.
# If you want to introduce a new group, ensure you also add it to .github/workflows/ci.yml
const GROUP = get(ENV, "GROUP", "")

# Writing tests:
# 1. The file structure of the test should match precisely the file structure of src.
#   Amongst other things, this means that there should be exactly 1 test file per src file.
#   This makes it trivially easy for someone to find the tests associated with a particular
#   src file.
# 2. A consequence of 1 is that there should be exactly 1 test file per src file.
# 3. A test file called foo.jl should have the structure:
#   @testset "foo" begin
#       code
#   end
#
#   Note that the testset is called `foo`, not `foo.jl`. Use whatever testset structure
#   seems appropriate within a given file. eg. if multiple types / functions are defined in
#   a particular source file, you might want multiple testsets in the test file.
# 4. Each directory should have its own testset, in which each test file is `include`d.
# 5. Each test file should create its own state, and shouldn't rely on state defined in
#   other test files. e.g. don't define a matrix used by all of the files in kernels. If
#   two test files are similar enough to share state, perhaps the corresponding source code
#   should be in the same file.
# 6. If you write a src file without any tests, create a corresponding test file with the
#   usual structure, but without any tests.
# 7. Explicitly create a new random number generate for _at_ _least_ each new test file, and
#   use it whenever generating randomness. This ensures complete control over random number
#   generation and makes it clear what randomness depends on other randomness.
# 8. All `using` statements should appear in runtests.jl.
# 9. List out all test files explicitly (eg. don't loop over them). This makes it easy to
#   disable tests by simply commenting them out, and makes it very clear which tests are not
#   currently being run.
# 10. If utility functionality is required, it should be placed in `src/test_utils.jl` so
#   that other packages can benefit from it when implementing new kernels.
@info "Packages Loaded"

include("test_utils.jl")

@testset "KernelFunctions" begin
    if GROUP == "" || GROUP == "Transform"
        @testset "transform" begin
            include(joinpath("transform", "transform.jl"))
            print(" ")
            include(joinpath("transform", "scaletransform.jl"))
            print(" ")
            include(joinpath("transform", "ardtransform.jl"))
            print(" ")
            include(joinpath("transform", "lineartransform.jl"))
            print(" ")
            include(joinpath("transform", "functiontransform.jl"))
            print(" ")
            include(joinpath("transform", "selecttransform.jl"))
            print(" ")
            include(joinpath("transform", "chaintransform.jl"))
            print(" ")
            include(joinpath("transform", "periodic_transform.jl"))
            print(" ")
            include(joinpath("transform", "with_lengthscale.jl"))
            print(" ")
        end
        @info "Ran tests on Transform"
    end

    if GROUP == "" || GROUP == "BaseKernels"
        @testset "basekernels" begin
            include(joinpath("basekernels", "constant.jl"))
            print(" ")
            include(joinpath("basekernels", "cosine.jl"))
            print(" ")
            include(joinpath("basekernels", "exponential.jl"))
            print(" ")
            include(joinpath("basekernels", "exponentiated.jl"))
            print(" ")
            include(joinpath("basekernels", "fbm.jl"))
            print(" ")
            include(joinpath("basekernels", "gabor.jl"))
            print(" ")
            include(joinpath("basekernels", "matern.jl"))
            print(" ")
            include(joinpath("basekernels", "nn.jl"))
            print(" ")
            include(joinpath("basekernels", "periodic.jl"))
            print(" ")
            include(joinpath("basekernels", "piecewisepolynomial.jl"))
            print(" ")
            include(joinpath("basekernels", "polynomial.jl"))
            print(" ")
            include(joinpath("basekernels", "rational.jl"))
            print(" ")
            include(joinpath("basekernels", "sm.jl"))
            print(" ")
            include(joinpath("basekernels", "wiener.jl"))
            print(" ")
        end
        @info "Ran tests on BaseKernel"
    end

    if GROUP == "" || GROUP == "Kernels"
        @testset "kernels" begin
            include(joinpath("kernels", "kernelproduct.jl"))
            include(joinpath("kernels", "kernelsum.jl"))
            include(joinpath("kernels", "kerneltensorproduct.jl"))
            include(joinpath("kernels", "overloads.jl"))
            include(joinpath("kernels", "scaledkernel.jl"))
            include(joinpath("kernels", "transformedkernel.jl"))
            include(joinpath("kernels", "normalizedkernel.jl"))
            include(joinpath("kernels", "neuralkernelnetwork.jl"))
        end
        @info "Ran tests on Kernel"
    end

    if GROUP == "" || GROUP == "MultiOutput"
        @testset "multi_output" begin
            include(joinpath("mokernels", "moinput.jl"))
            include(joinpath("mokernels", "independent.jl"))
            include(joinpath("mokernels", "slfm.jl"))
            include(joinpath("mokernels", "intrinsiccoregion.jl"))
            include(joinpath("mokernels", "lmm.jl"))
        end
        @info "Ran tests on Multi-Output Kernels"
    end

    if GROUP == "" || GROUP == "Others"
        include("utils.jl")

        @testset "distances" begin
            include(joinpath("distances", "pairwise.jl"))
            include(joinpath("distances", "dotproduct.jl"))
            include(joinpath("distances", "delta.jl"))
            include(joinpath("distances", "sinus.jl"))
        end
        @info "Ran tests on Distances"

        @testset "matrix" begin
            include(joinpath("matrix", "kernelmatrix.jl"))
            include(joinpath("matrix", "kernelkroneckermat.jl"))
            include(joinpath("matrix", "kernelpdmat.jl"))
        end
        @info "Ran tests on matrix"

        @testset "approximations" begin
            include(joinpath("approximations", "nystrom.jl"))
        end

        include("generic.jl")
        include("chainrules.jl")
        include("zygoterules.jl")

        @testset "doctests" begin
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
            doctest(
                KernelFunctions;
                doctestfilters=[
                    r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
                    r"(\s?Array{[a-zA-Z0-9]+,\s?1}|\s?Vector{[a-zA-Z0-9]+})",
                    r"(\s?Array{[a-zA-Z0-9]+,\s?2}|\s?Matrix{[a-zA-Z0-9]+})",
                ],
            )
        end
    end
end

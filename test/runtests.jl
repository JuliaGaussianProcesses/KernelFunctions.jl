using SafeTestsets

using Test

# Writing tests:
# 1. The file structure of the test should match precisely the file structure of src.
#   Amongst other things, this means that there should be exactly 1 test file per src file.
#   This makes it trivially easy for someone to find the tests associated with a particular
#   src file.
# 2. A consequence of 1 is that there should be exactly 1 test file per src file.
# 3. Use whatever testset structure seems appropriate within a given file. eg. if multiple
#   types / functions are defined in a particular source file, you might want multiple
#   testsets in the test file.
# 4. Each directory should have its own testset, in which each test file called `foo.jl` is
#   `include`d in a `@safetestset` called "foo":
#   @safetestset "foo" begin include(joinpath(..., "foo.jl")) end
#
#   Note that the testset is called `foo`, not `foo.jl`.
# 5. Each test file creates its own state, and must not rely on state defined in
#   other test files. e.g. don't define a matrix used by all of the files in kernels. If
#   two test files are similar enough to share state, perhaps the corresponding source code
#   should be in the same file.
# 6. If you write a src file without any tests, create a corresponding test file with the
#   usual structure, but without any tests.
# 7. Explicitly create a new random number generate for _at_ _least_ each new test file, and
#   use it whenever generating randomness. This ensures complete control over random number
#   generation and makes it clear what randomness depends on other randomness.
# 8. All `using` statements should appear in the test files, only `SafeTestsets` and `Test`
#   should be imported in runtests.jl.
# 9. List out all test files explicitly (eg. don't loop over them). This makes it easy to
#   disable tests by simply commenting them out, and makes it very clear which tests are not
#   currently being run.
# 10. If utility files are required.

@testset "KernelFunctions" begin

    @safetestset "utils" begin include("utils.jl") end

    @testset "distances" begin
        @safetestset "dotproduct" begin include(joinpath("distances", "dotproduct.jl")) end
        @safetestset "delta" begin include(joinpath("distances", "delta.jl")) end
    end

    @testset "transform" begin
        @safetestset "scaletransform" begin
            include(joinpath("transform", "scaletransform.jl"))
        end
        @safetestset "ardtransform" begin
            include(joinpath("transform", "ardtransform.jl"))
        end
        @safetestset "lowranktransform" begin
            include(joinpath("transform", "lowranktransform.jl"))
        end
        @safetestset "functiontransform" begin
            include(joinpath("transform", "functiontransform.jl"))
        end
        @safetestset "selecttransform" begin
            include(joinpath("transform", "selecttransform.jl"))
        end
        @safetestset "chaintransform" begin
            include(joinpath("transform", "chaintransform.jl"))
        end
        @safetestset "transform" begin include(joinpath("transform", "transform.jl")) end
    end

    @testset "kernels" begin
        @safetestset "constant" begin include(joinpath("kernels", "constant.jl")) end
        @safetestset "cosine" begin include(joinpath("kernels", "cosine.jl")) end
        @safetestset "exponential" begin include(joinpath("kernels", "exponential.jl")) end
        @safetestset "exponentiated" begin
            include(joinpath("kernels", "exponentiated.jl"))
        end
        @safetestset "fbm" begin include(joinpath("kernels", "fbm.jl")) end
    @safetestset "gabor" begin include(joinpath("kernels", "gabor.jl")) end
        @safetestset "kernelproduct" begin
            include(joinpath("kernels", "kernelproduct.jl"))
        end
        @safetestset "kernelsum" begin include(joinpath("kernels", "kernelsum.jl")) end
        @safetestset "maha" begin include(joinpath("kernels", "maha.jl")) end
        @safetestset "matern" begin include(joinpath("kernels", "matern.jl")) end
        @safetestset "polynomial" begin include(joinpath("kernels", "polynomial.jl")) end
        @safetestset "rationalquad" begin
            include(joinpath("kernels", "rationalquad.jl"))
        end
        @safetestset "scaledkernel" begin
            include(joinpath("kernels", "scaledkernel.jl"))
        end
        @safetestset "transformedkernel" begin
            include(joinpath("kernels", "transformedkernel.jl"))
        end

        # Legacy tests that don't correspond to anything meaningful in src. Unclear how
        # helpful these are.
        @safetestset "custom" begin include(joinpath("kernels", "custom.jl")) end
    end

    @testset "matrix" begin
        @safetestset "kernelmatrix" begin
            include(joinpath("matrix", "kernelmatrix.jl"))
        end
        @safetestset "kernelkroneckermat" begin
            include(joinpath("matrix", "kernelkroneckermat.jl"))
        end
        @safetestset "kernelpdmat" begin include(joinpath("matrix", "kernelpdmat.jl")) end
    end

    @testset "approximations" begin
        @safetestset "nystrom" begin include(joinpath("approximations", "nystrom.jl")) end
    end

    @safetestset "generic" begin include("generic.jl") end
    @safetestset "zygote_adjoints" begin include("zygote_adjoints.jl") end
    @safetestset "trainable" begin include("trainable.jl") end
end

# These are legacy tests that I'm not getting rid of, as they appear to be useful, but
# weren't enabled on master at the time of refactoring the tests. They will need to be
# restored at some point.
# include("utils_AD.jl")
# include("test_AD.jl")

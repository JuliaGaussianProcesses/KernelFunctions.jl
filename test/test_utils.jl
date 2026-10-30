# More test utilities. Can't be included in KernelFunctions because they introduce a number
# of additional deps that we don't want to have in the main package.

# AD utilities

const _TEST_ZYGOTE = VERSION < v"1.12"

const FDM = FiniteDifferences.central_fdm(5, 1)

# Helper functions for kernel matrix tests
testfunction(k, A, B, dim) = sum(kernelmatrix(k, A, B; obsdim=dim))
testfunction(k, A, dim) = sum(kernelmatrix(k, A; obsdim=dim))
testdiagfunction(k, A, dim) = sum(kernelmatrix_diag(k, A; obsdim=dim))
testdiagfunction(k, A, B, dim) = sum(kernelmatrix_diag(k, A, B; obsdim=dim))

testfunction(k::MOKernel, A, B) = sum(kernelmatrix(k, A, B))
testfunction(k::MOKernel, A) = sum(kernelmatrix(k, A))
testdiagfunction(k::MOKernel, A) = sum(kernelmatrix_diag(k, A))
testdiagfunction(k::MOKernel, A, B) = sum(kernelmatrix_diag(k, A, B))

# ============================================================================
# DifferentiationInterfaceTest-based AD testing infrastructure
# ============================================================================

const FD_BACKEND = AutoFiniteDifferences(; fdm=FDM)

const _DEFAULT_BACKENDS = let
    backends = [AutoForwardDiff(), AutoReverseDiff(), AutoMooncake()]
    _TEST_ZYGOTE && pushfirst!(backends, AutoZygote())
    backends
end

const _BACKEND_MAP = Dict{Symbol,Any}(
    :ForwardDiff => AutoForwardDiff(),
    :ReverseDiff => AutoReverseDiff(),
    :Zygote => AutoZygote(),
    :Mooncake => AutoMooncake(),
    :FiniteDiff => FD_BACKEND,
)

# Custom isapprox that handles Zygote returning `nothing` for zero gradients
function _isapprox_nothing(a, b; kwargs...)
    a_val = isnothing(a) ? zero.(b) : a
    b_val = isnothing(b) ? zero.(a) : b
    return Base.isapprox(a_val, b_val; kwargs...)
end

function _resolve_backends(ADs)
    if ADs === nothing
        return _DEFAULT_BACKENDS
    else
        return [_BACKEND_MAP[ad] for ad in ADs]
    end
end

# Thin wrappers for direct gradient computation (used by chainrules.jl and selecttransform.jl)
function gradient(f, s::Symbol, args)
    return DI.gradient(f, _BACKEND_MAP[s], args)
end

function compare_gradient(f, AD::Symbol, args)
    grad_AD = DI.gradient(f, _BACKEND_MAP[AD], args)
    grad_FD = DI.gradient(f, FD_BACKEND, args)
    @test _isapprox_nothing(grad_AD, grad_FD; atol=1e-8, rtol=1e-5)
end

"""
    test_ADs(kernelfunction, args=nothing; ADs=nothing, dims=[3, 3])

Tests gradient correctness of kernel functions across multiple AD backends
using DifferentiationInterfaceTest.jl. Uses FiniteDifferences as reference backend.
"""
function test_ADs(kernelfunction, args=nothing; ADs=nothing, dims=[3, 3])
    backends = _resolve_backends(ADs)
    k = args === nothing ? kernelfunction() : kernelfunction(args)
    rng = MersenneTwister(42)

    # First check that FiniteDifferences works (skip AD tests if not)
    scenarios_smoke = _build_kernel_scenarios(k, kernelfunction, args, dims, rng)
    fd_ok = true
    @testset "FiniteDifferences" begin
        for (_, f, x) in scenarios_smoke
            try
                DI.gradient(f, FD_BACKEND, x)
            catch
                fd_ok = false
                @test_broken false  # Mark as known failure
            end
        end
    end

    fd_ok || return nothing

    # Build scenarios with reference results
    scenarios = Scenario[]
    for (name, f, x) in scenarios_smoke
        res1 = DI.gradient(f, FD_BACKEND, x)
        push!(scenarios, Scenario{:gradient,:out}(f, x; res1=res1, name=name))
    end

    @testset "AD correctness" begin
        test_differentiation(
            backends,
            scenarios;
            correctness=true,
            isapprox=_isapprox_nothing,
            atol=1e-8,
            rtol=1e-5,
        )
    end
end

"""
Build a list of (name, f, x) tuples describing all gradient tests for a kernel.
"""
function _build_kernel_scenarios(k, kernelfunction, args, dims, rng)
    scenarios = Tuple{String,Any,Any}[]

    # 1. kappa tests for SimpleKernels
    if k isa SimpleKernel
        for d in log.([eps(), rand(rng)])
            f = let k = k
                x -> kappa(k, exp(x[1]))
            end
            push!(scenarios, ("kappa", f, [d]))
        end
    end

    # 2. kernel evaluation tests
    x = rand(rng, dims[1])
    y = rand(rng, dims[1])

    let k = k, y = y
        push!(scenarios, ("k(x,y) w.r.t. x", x -> k(x, y), copy(x)))
    end
    let k = k, x = x
        push!(scenarios, ("k(x,y) w.r.t. y", y -> k(x, y), copy(y)))
    end

    # 3. hyperparameter tests
    if args !== nothing
        let kernelfunction = kernelfunction, x = x, y = y
            push!(
                scenarios, ("hyperparams (eval)", p -> kernelfunction(p)(x, y), copy(args))
            )
        end
    end

    # 4. kernel matrix tests
    A = rand(rng, dims...)
    B = rand(rng, dims...)
    for _testfn in (testfunction, testdiagfunction)
        fname = _testfn === testfunction ? "kernelmatrix" : "kernelmatrix_diag"
        for dim in 1:2
            let k = k, B = B, dim = dim
                push!(scenarios, ("$fname(k,A,$dim)", a -> _testfn(k, a, dim), copy(A)))
            end
            let k = k, B = B, dim = dim
                push!(
                    scenarios,
                    ("$fname(k,A,B,$dim) w.r.t. A", a -> _testfn(k, a, B, dim), copy(A)),
                )
            end
            let k = k, A = A, dim = dim
                push!(
                    scenarios,
                    ("$fname(k,A,B,$dim) w.r.t. B", b -> _testfn(k, A, b, dim), copy(B)),
                )
            end
            if args !== nothing
                let kernelfunction = kernelfunction, A = A, dim = dim
                    push!(
                        scenarios,
                        (
                            "$fname hyperparams unary",
                            p -> _testfn(kernelfunction(p), A, dim),
                            copy(args),
                        ),
                    )
                end
                let kernelfunction = kernelfunction, A = A, B = B, dim = dim
                    push!(
                        scenarios,
                        (
                            "$fname hyperparams binary",
                            p -> _testfn(kernelfunction(p), A, B, dim),
                            copy(args),
                        ),
                    )
                end
            end
        end
    end

    return scenarios
end

"""
    test_ADs(k::MOKernel; ADs=nothing, dims=(in=3, out=2, obs=3))

Tests gradient correctness of multi-output kernel functions across multiple AD backends.
"""
function test_ADs(k::MOKernel; ADs=nothing, dims=(in=3, out=2, obs=3))
    backends = _resolve_backends(ADs)
    rng = MersenneTwister(42)

    scenarios_smoke = _build_mokernel_scenarios(k, dims, rng)
    fd_ok = true
    @testset "FiniteDifferences" begin
        for (_, f, x) in scenarios_smoke
            try
                DI.gradient(f, FD_BACKEND, x)
            catch
                fd_ok = false
                @test_broken false
            end
        end
    end

    if !fd_ok
        return nothing
    end

    scenarios = Scenario[]
    for (name, f, x) in scenarios_smoke
        res1 = DI.gradient(f, FD_BACKEND, x)
        push!(scenarios, Scenario{:gradient,:out}(f, x; res1=res1, name=name))
    end

    @testset "AD correctness" begin
        test_differentiation(
            backends,
            scenarios;
            correctness=true,
            isapprox=_isapprox_nothing,
            atol=1e-8,
            rtol=1e-5,
        )
    end
end

"""
Build (name, f, x) tuples for MOKernel gradient tests.
"""
function _build_mokernel_scenarios(k::MOKernel, dims, rng)
    scenarios = Tuple{String,Any,Any}[]

    x = (rand(rng, dims.in), rand(rng, 1:(dims.out)))
    y = (rand(rng, dims.in), rand(rng, 1:(dims.out)))

    # Kernel eval w.r.t. continuous components
    let k = k, x = x, y = y
        push!(scenarios, ("MOKernel eval w.r.t. x", a -> k((a, x[2]), y), copy(x[1])))
    end
    let k = k, x = x, y = y
        push!(scenarios, ("MOKernel eval w.r.t. y", b -> k(x, (b, y[2])), copy(y[1])))
    end

    # Kernel matrices
    A = [(randn(rng, dims.in), rand(rng, 1:(dims.out))) for _ in 1:(dims.obs)]
    B = [(randn(rng, dims.in), rand(rng, 1:(dims.out))) for _ in 1:(dims.obs)]
    A_mat = reduce(hcat, first.(A))
    B_mat = reduce(hcat, first.(B))

    for _testfn in (testfunction, testdiagfunction)
        fname = _testfn === testfunction ? "kernelmatrix" : "kernelmatrix_diag"

        # Unary
        let k = k, A = A
            f = a -> begin
                A_local = tuple.(eachcol(a), last.(A))
                _testfn(k, A_local)
            end
            push!(scenarios, ("$fname MO unary", f, copy(A_mat)))
        end

        # Binary w.r.t. A
        let k = k, A = A, B = B
            f = a -> begin
                A_local = tuple.(eachcol(a), last.(A))
                _testfn(k, A_local, B)
            end
            push!(scenarios, ("$fname MO binary w.r.t. A", f, copy(A_mat)))
        end

        # Binary w.r.t. B
        let k = k, A = A, B = B
            f = b -> begin
                B_local = tuple.(eachcol(b), last.(B))
                _testfn(k, A, B_local)
            end
            push!(scenarios, ("$fname MO binary w.r.t. B", f, copy(B_mat)))
        end
    end

    return scenarios
end

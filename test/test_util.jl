using FiniteDifferences: j′vp

# Default tolerances for testing Zygote against FiniteDifferences.
const _rtol = 1e-10
const _atol = 1e-10



#
# Print stuff for debugging
#

function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    @show typeof(adjoint_ad), typeof(adjoint_fd)
    adjoint_ad, adjoint_fd = to_vec(adjoint_ad)[1], to_vec(adjoint_fd)[1]
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs.((adjoint_ad .- adjoint_fd) ./ adjoint_ad)
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end



#
# Version of isapprox that works for lots of types.
#

function fd_isapprox(x_ad::Nothing, x_fd, rtol, atol)
    return fd_isapprox(x_fd, zero(x_fd), rtol, atol)
end
function fd_isapprox(x_ad::AbstractArray, x_fd::AbstractArray, rtol, atol)
    return all(fd_isapprox.(x_ad, x_fd, rtol, atol))
end
function fd_isapprox(x_ad::Real, x_fd::Real, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::NamedTuple, x_fd, rtol, atol)
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′)->fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end
function fd_isapprox(x_ad::Dict, x_fd::Dict, rtol, atol)
    return all([fd_isapprox(get(()->nothing, x_ad, key), x_fd[key], rtol, atol) for
        key in keys(x_fd)])
end



#
# Check Zygote against FiniteDifferences.
#

function adjoint_test(
    f, ȳ, x...;
    rtol=_rtol,
    atol=_atol,
    fdm=FiniteDifferences.Central(5, 1),
    print_results=false,
)

    # Compute forwards-pass and j′vp.
    y, back = Zygote.forward(f, x...)
    adj_ad = back(ȳ)
    adj_fd = j′vp(fdm, f, ȳ, x...)

    # If unary, pull out first thing from ad.
    adj_ad = length(x) == 1 ? first(adj_ad) : adj_ad

    # Check that forwards-pass agrees with plain forwards-pass.
    @test y ≈ f(x...)

    # Check that ad and fd adjoints (approximately) agree.
    print_results && print_adjoints(adj_ad, adj_fd, rtol, atol)
    @test fd_isapprox(adj_ad, adj_fd, rtol, atol)
end

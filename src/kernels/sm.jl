"""
    SpectralMixtureKernel()

Gaussian Spectral Mixture kernel function. The kernel function
parametrization depends on the sign of Q.

Let `t(Dx1)` be an offset vector in dataspace e.g. `t = x-z`. Then `w(DxP)`
are the weights and `m(Dx|Q|) = 1/p`, `v(Dx|Q|) = (2*pi*ell)^-2` are spectral
means (frequencies) and variances, where `p` is the period and `ell` the length
scale of the Gabor function `h(t2v,tm)` given by the expression

```julia
    h(t2v, tm) = exp(-2 * pi^2 * t2v) .* cos(2 * pi * tm)
```

Then, the two covariances are obtained as follows:

 SM, spectral mixture:          Q>0 => P = 1
```julia
   k(x, y) = w' * h((t .* t)' * v, t' * m), t = x-y
```

 SMP, spectral mixture product: Q<0 => P = D
```julia
   k(x, y) = prod(w' * h(T * T * v, T * m)), T = diag(t), t = x-y
```

Note that for D=1, the two modes +Q and -Q are exactly the same.

**References:**\\
    [1] SM: Gaussian Process Kernels for Pattern Discovery and Extrapolation,
        ICML, 2013, by Andrew Gordon Wilson and Ryan Prescott Adams,
    [2] SMP: GPatt: Fast Multidimensional Pattern Extrapolation with GPs,
        arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa,
        Arye Nehorai and John P. Cunningham, and
    [3] Covariance kernels for fast automatic pattern discovery and extrapolation
        with Gaussian processes, Andrew Gordon Wilson, PhD Thesis, January 2014.
        http://www.cs.cmu.edu/~andrewgw/andrewgwthesis.pdf
    [4] http://www.cs.cmu.edu/~andrewgw/pattern/.

"""
struct SpectralMixtureKernel{
    K<:Kernel,
    M1<:AbstractMatrix{<:Real},
    M2<:AbstractMatrix{<:Real},
    M3<:AbstractMatrix{<:Real}} <: BaseKernel
    w::M1
    m::M2
    v::M3
    kernel::K
    function SpectralMixtureKernel(;w , m , v)
        size(m) == size(v) || error("Dimensions of means m and variances v do not match.")
        size(w, 1) == size(m, 1) == size(v, 1) ||
            error("First dimension of weights w, means m, variances v are does not match.")
        k = GaborKernel(ell= 1 / (2 * pi^2), p= 1 / 2)
        new{typeof(k), typeof(w),typeof(m),typeof(v)}(w, m, v, k)
    end
end

function kappa(
    κ::SpectralMixtureKernel,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real})

    D = size(κ.w, 1)
    P = size(κ.w, 2)
    Q = size(κ.m, 2)
    t = x - y
    @info size(t')
    @info size((t.^2)' * κ.v)
    @info size(t' * κ.m)
    @info size(map(d -> kappa(κ.kernel.kernel.kernels[1], d), (t.^2)' * κ.v))
    @info size(map(d -> kappa(κ.kernel.kernel.kernels[2], d), t' * κ.m))
    @info size(κ.w')

    # w' * h((t.*t)'*v,t'*m)
    # return w' * map(d -> kappa(κ.kernel.kernels[1], d), (t.^2)' * κ.v) *
    #     map(d -> kappa(κ.kernel.kernels[2], d), dot(t, m))
end

Base.show(io::IO, κ::SpectralMixtureKernel) = print(io, "Spectral Mixture Kernel")

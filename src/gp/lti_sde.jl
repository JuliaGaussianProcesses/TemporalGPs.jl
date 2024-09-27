"""
    LTISDE (Linear Time-Invariant Stochastic Differential Equation)

A lightweight wrapper around a `GP` `f` that tells this package to handle inference in `f`.
Can be constructed via the [`to_sde`](@ref) function.
"""
struct LTISDE{Tf<:GP, Tstorage<:StorageType} <: AbstractGP
    f::Tf
    storage::Tstorage
end

function to_sde(f::GP, storage_type=ArrayStorage(Float64))
    return LTISDE(f, storage_type)
end

storage_type(f::LTISDE) = f.storage

"""
    const FiniteLTISDE = FiniteGP{<:LTISDE}

A `FiniteLTISDE` is just a regular `FiniteGP` that happens to contain an `LTISDE`, as
opposed to any other `AbstractGP`, useful for dispatching.
"""
const FiniteLTISDE = FiniteGP{<:LTISDE}

# Deal with a bug in AbstractGPs.
function AbstractGPs.FiniteGP(f::LTISDE, x::AbstractVector{<:Real})
    return AbstractGPs.FiniteGP(f, x, convert(eltype(storage_type(f)), 1e-12))
end

# Implement the AbstractGP API.

function AbstractGPs.marginals(ft::FiniteLTISDE)
    return reduce(vcat, map(marginals, marginals(build_lgssm(ft))))
end

function AbstractGPs.mean_and_var(ft::FiniteLTISDE)
    ms = marginals(ft)
    return map(mean, ms), map(var, ms)
end

AbstractGPs.mean(ft::FiniteLTISDE) = first(mean_and_var(ft))

AbstractGPs.var(ft::FiniteLTISDE) = last(mean_and_var(ft))

AbstractGPs.cov(ft::FiniteLTISDE) = cov(FiniteGP(ft.f.f, ft.x, ft.Σy))

function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteLTISDE)
    return destructure(ft.x, rand(rng, build_lgssm(ft)))
end

AbstractGPs.rand(ft::FiniteLTISDE) = rand(Random.GLOBAL_RNG, ft)

function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteLTISDE, N::Int)
    return hcat([rand(rng, ft) for _ in 1:N]...)
end

AbstractGPs.rand(ft::FiniteLTISDE, N::Int) = rand(Random.GLOBAL_RNG, ft, N)

AbstractGPs.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Real}) = _logpdf(ft, y)

function AbstractGPs.logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}})
    return _logpdf(ft, y)
end

function _logpdf(ft::FiniteLTISDE, y::AbstractVector{<:Union{Missing, Real}})
    return logpdf(build_lgssm(ft), observations_to_time_form(ft.x, y))
end

# Converting GPs into LGSSMs (Linear Gaussian State-Space Models).
function build_lgssm(f::LTISDE, x::AbstractVector, Σys::AbstractVector)
    m = get_mean(f)
    k = get_kernel(f)
    As, as, Qs, emission_proj, x0 = lgssm_components(m, k, x, f.storage)
    return LGSSM(
        GaussMarkovModel(Forward(), As, as, Qs, x0), build_emissions(emission_proj, Σys),
    )
end

build_lgssm(ft::FiniteLTISDE) = build_lgssm(ft.f, ft.x, noise_var_to_time_form(ft.x, ft.Σy))

get_mean(f::LTISDE) = get_mean(f.f)
get_mean(f::GP) = f.mean

get_kernel(f::LTISDE) = get_kernel(f.f)
get_kernel(f::GP) = f.kernel

function build_emissions(
    (Hs, hs)::Tuple{AbstractVector, AbstractVector}, Σs::AbstractVector,
)
    Hst = map(adjoint, Hs)
    return StructArray{get_type(Hst, hs, Σs)}((Hst, hs, Σs))
end

function get_type(Hs_prime, hs::AbstractVector{<:Real}, Σs)
    THs = eltype(Hs_prime)
    Ths = eltype(hs)
    TΣs = eltype(Σs)
    T = ScalarOutputLGC{THs, Ths, TΣs}
    return T
end

function get_type(Hs_prime, hs::AbstractVector{<:AbstractVector}, Σs)
    THs = eltype(Hs_prime)
    Ths = eltype(hs)
    TΣs = eltype(Σs)
    T = SmallOutputLGC{THs, Ths, TΣs}
    return T
end

# Constructor for combining kernel and mean functions
function lgssm_components(
    ::ZeroMean, k::Kernel, t::AbstractVector, storage_type::StorageType
)
    return lgssm_components(k, t, storage_type)
end

function lgssm_components(
    m::AbstractGPs.MeanFunction, k::Kernel, t::AbstractVector, storage_type::StorageType
)
    m = collect(mean_vector(m, t)) # `collect` is needed as there are still issues with Zygote and FillArrays.
    As, as, Qs, (Hs, hs), x0 = lgssm_components(k, t, storage_type)
    hs = add_proj_mean(hs, m)
    return As, as, Qs, (Hs, hs), x0
end

# Either build a new vector or update an existing one with 
add_proj_mean(hs::AbstractVector{<:Real}, m) = hs .+ m
function add_proj_mean(hs::AbstractVector, m)
    return map((h, m) -> h + vcat(m, Zeros(length(h) - 1)), hs, m)
end

# Really just a hook for AD.
time_exp(A, t) = exp(A * t)

# Generic constructors for base kernels.

function broadcast_components(
    (F, q, H)::Tuple, x0::Gaussian, t::AbstractVector{<:Real}, ::StorageType{T}
) where {T}
    P = Symmetric(x0.P)
    t = vcat([first(t) - 1], t)
    As = map(Δt -> time_exp(F, T(Δt)), diff(t))
    as = Fill(Zeros{T}(size(first(As), 1)), length(As))
    Qs = map(A -> P - A * P * A', As)
    Hs = Fill(H, length(As))
    hs = Fill(zero(T), length(As))
    As, as, Qs, Hs, hs
end

function broadcast_components(
    (F, q, H)::Tuple, x0::Gaussian, t::Union{StepRangeLen, RegularSpacing}, ::StorageType{T}
) where {T}
    P = Symmetric(x0.P)
    A = time_exp(F, T(step(t)))
    As = Fill(A, length(t))
    as = Fill(Zeros{T}(size(F, 1)), length(t))
    Q = Symmetric(P) - A * Symmetric(P) * A'
    Qs = Fill(Q, length(t))
    Hs = Fill(H, length(t))
    hs = Fill(zero(T), length(As))
    As, as, Qs, Hs, hs
end

function lgssm_components(
    k::SimpleKernel, t::AbstractVector{<:Real}, storage::StorageType{T},
) where {T<:Real}

    # Compute stationary distribution and sde.
    x0 = stationary_distribution(k, storage)
    # Use stationary distribution + sde to compute finite-dimensional Gauss-Markov model.
    As, as, Qs, Hs, hs = broadcast_components(to_sde(k, storage), x0, t, storage)
    
    emission_projections = (Hs, hs)

    return As, as, Qs, emission_projections, x0
end

# Fallback definitions for most base kernels.
function to_sde(k::SimpleKernel, ::ArrayStorage{T}) where {T<:Real}
    F, q, H = to_sde(k, SArrayStorage(T))
    return collect(F), q, collect(H)
end

function stationary_distribution(k::SimpleKernel, ::ArrayStorage{T}) where {T<:Real}
    x = stationary_distribution(k, SArrayStorage(T))
    return Gaussian(collect(x.m), collect(x.P))
end

safe_to_product(::Kernel) = false

# Matern-1/2

function to_sde(::Matern12Kernel, ::SArrayStorage{T}) where {T<:Real}
    F = SMatrix{1, 1, T}(-1)
    q = convert(T, 2)
    H = SVector{1, T}(1)
    return F, q, H
end

function stationary_distribution(::Matern12Kernel, ::SArrayStorage{T}) where {T<:Real}
    return Gaussian(
        SVector{1, T}(0),
        SMatrix{1, 1, T}(1),
    )
end

safe_to_product(::Matern12Kernel) = true

# Matern - 3/2

function to_sde(::Matern32Kernel, ::SArrayStorage{T}) where {T<:Real}
    λ = sqrt(3)
    F = SMatrix{2, 2, T}(0, -3, 1, -2λ)
    q = convert(T, 4 * λ^3)
    H = SVector{2, T}(1, 0)
    return F, q, H
end

function stationary_distribution(::Matern32Kernel, ::SArrayStorage{T}) where {T<:Real}
    return Gaussian(
        SVector{2, T}(0, 0),
        SMatrix{2, 2, T}(1, 0, 0, 3),
    )
end

safe_to_product(::Matern32Kernel) = true

# Matern - 5/2

function to_sde(::Matern52Kernel, ::SArrayStorage{T}) where {T<:Real}
    λ = sqrt(5)
    F = SMatrix{3, 3, T}(0, 0, -λ^3, 1, 0, -3λ^2, 0, 1, -3λ)
    q = convert(T, 8 * λ^5 / 3)
    H = SVector{3, T}(1, 0, 0)
    return F, q, H
end

function stationary_distribution(::Matern52Kernel, ::SArrayStorage{T}) where {T<:Real}
    κ = 5 / 3
    m = SVector{3, T}(0, 0, 0)
    P = SMatrix{3, 3, T}(1, 0, -κ, 0, κ, 0, -κ, 0, 25)
    return Gaussian(m, P)
end

safe_to_product(::Matern52Kernel) = true

# Cosine

function to_sde(::CosineKernel, ::SArrayStorage{T}) where {T}
    F = SMatrix{2, 2, T}(0, 1, -1, 0)
    q = zero(T)
    H = SVector{2, T}(1, 0)
    return F, q, H
end
    
function stationary_distribution(::CosineKernel, ::SArrayStorage{T}) where {T<:Real}
    m = SVector{2, T}(0, 0)
    P = SMatrix{2, 2, T}(1, 0, 0, 1)
    return Gaussian(m, P)
end

safe_to_product(::CosineKernel) = true

# ApproxPeriodicKernel

# The periodic kernel is approximated by a sum of cosine kernels with different frequencies.
struct ApproxPeriodicKernel{N,K<:PeriodicKernel} <: KernelFunctions.SimpleKernel
    kernel::K
    function ApproxPeriodicKernel{N,K}(kernel::K) where {N,K<:PeriodicKernel}
        length(kernel.r) == 1 || error("ApproxPeriodicKernel only supports a single lengthscale")
        return new{N,K}(kernel)
    end
end
# We follow "State Space approximation of Gaussian Processes for time series forecasting"
# by Alessio Benavoli and Giorgio Corani and use a default of 7 Cosine Kernel terms
ApproxPeriodicKernel(;r::Real=1.0) = ApproxPeriodicKernel{7}(PeriodicKernel(;r=[r]))
ApproxPeriodicKernel{N}(;r::Real=1.0) where {N} = ApproxPeriodicKernel{N}(PeriodicKernel(;r=[r]))
ApproxPeriodicKernel(kernel::PeriodicKernel) = ApproxPeriodicKernel{7}(kernel)
ApproxPeriodicKernel{N}(kernel::K) where {N,K<:PeriodicKernel} = ApproxPeriodicKernel{N,K}(kernel)

KernelFunctions.kappa(k::ApproxPeriodicKernel, x) = KernelFunctions.kappa(k.kernel, x)
KernelFunctions.metric(k::ApproxPeriodicKernel) = KernelFunctions.metric(k.kernel)

function Base.show(io::IO, κ::ApproxPeriodicKernel{N}) where {N}
    return print(io, "Approximate Periodic Kernel, (r = $(only(κ.kernel.r))) approximated with $N cosine kernels")
end

# Can't use approx periodic kernel with static arrays -- the dimensions become too large.
_ap_error() = throw(error("Unable to construct an ApproxPeriodicKernel for SArrayStorage"))
to_sde(::ApproxPeriodicKernel, ::SArrayStorage) = _ap_error()
stationary_distribution(::ApproxPeriodicKernel, ::SArrayStorage) = _ap_error()

function to_sde(::ApproxPeriodicKernel{N}, storage::ArrayStorage{T}) where {T<:Real, N}

    # Compute F and H for component processes.
    F, _, H = to_sde(CosineKernel(), storage)
    Fs = ntuple(N) do i
        2π * (i - 1) * F
    end

    # Combine component processes into a single whole.
    F = block_diagonal(collect.(Fs)...)
    q = zero(T)
    H = repeat(collect(H), N)
    return F, q, H
end

function stationary_distribution(kernel::ApproxPeriodicKernel{N}, storage::ArrayStorage{<:Real}) where {N}
    x0 = stationary_distribution(CosineKernel(), storage)
    m = collect(repeat(x0.m, N))
    r = kernel.kernel.r
    l⁻² = inv(4 * only(r)^2)
    Ps = ntuple(N) do j
        qⱼ = (1 + (j !== 1) ) * besseli(j - 1, l⁻²) / exp(l⁻²)
        return qⱼ * x0.P
    end
    P = collect(block_diagonal(Ps...))
    return Gaussian(m, P)
end

safe_to_product(::ApproxPeriodicKernel) = true

# Constant

function TemporalGPs.to_sde(::ConstantKernel, ::SArrayStorage{T}) where {T<:Real}
    F = SMatrix{1, 1, T}(0)
    q = convert(T, 0)
    H = SVector{1, T}(1)
    return F, q, H
end

function TemporalGPs.stationary_distribution(k::ConstantKernel, ::SArrayStorage{T}) where {T<:Real}
    return TemporalGPs.Gaussian(SVector{1, T}(0), SMatrix{1, 1, T}(T(only(k.c))))
end

safe_to_product(::ConstantKernel) = true


# Scaled

function to_sde(k::ScaledKernel, storage::StorageType{T}) where {T<:Real}
    F, q, H = to_sde(k.kernel, storage)
    σ = sqrt(convert(eltype(storage), only(k.σ²)))
    return F, σ^2 * q, σ * H
end

function stationary_distribution(k::ScaledKernel, storage::StorageType)
    return stationary_distribution(k.kernel, storage)
end

safe_to_product(k::ScaledKernel) = safe_to_product(k.kernel)

function lgssm_components(k::ScaledKernel, ts::AbstractVector, storage_type::StorageType)
    As, as, Qs, emission_proj, x0 = lgssm_components(k.kernel, ts, storage_type)
    σ = sqrt(convert(eltype(storage_type), only(k.σ²)))
    return As, as, Qs, _scale_emission_projections(emission_proj, σ), x0
end

function _scale_emission_projections((Hs, hs)::Tuple{AbstractVector, AbstractVector}, σ::Real)
    return map(H->σ * H, Hs), map(h->σ * h, hs)
end

function _scale_emission_projections((Cs, cs, Hs, hs), σ)
    return (Cs, cs, map(H -> σ * H, Hs), map(h -> σ * h, hs))
end

# Stretched

function to_sde(k::TransformedKernel{<:Kernel, <:ScaleTransform}, storage::StorageType)
    F, q, H = to_sde(k.kernel, storage)
    return F * only(k.transform.s), q, H
end

function stationary_distribution(
    k::TransformedKernel{<:Kernel, <:ScaleTransform}, storage::StorageType
)
    return stationary_distribution(k.kernel, storage)
end

safe_to_product(::TransformedKernel{<:Kernel, <:ScaleTransform}) = false

function lgssm_components(
    k::TransformedKernel{<:Kernel, <:ScaleTransform},
    ts::AbstractVector,
    storage_type::StorageType,
)
    return lgssm_components(k.kernel, apply_stretch(k.transform.s[1], ts), storage_type)
end

apply_stretch(a, ts::AbstractVector{<:Real}) = a * ts

apply_stretch(a, ts::StepRangeLen) = a * ts

apply_stretch(a, ts::RegularSpacing) = RegularSpacing(a * ts.t0, a * ts.Δt, ts.N)

# Product

safe_to_product(k::KernelProduct) = all(safe_to_product, k.kernels)

function lgssm_components(k::KernelProduct, ts::AbstractVector, storage::StorageType)

    safe_to_product(k) || throw(ArgumentError("Not all kernels in k are safe to product."))

    sde_kernels = to_sde.(k.kernels, Ref(storage))
    F_kernels = getindex.(sde_kernels, 1)
    F = foldl(_kron_add, F_kernels)
    q_kernels = getindex.(sde_kernels, 2)
    q = kron(q_kernels...)
    H_kernels = getindex.(sde_kernels, 3)
    H = kron(H_kernels...)

    x0_kernels = stationary_distribution.(k.kernels, Ref(storage))
    m_kernels = getproperty.(x0_kernels, :m)
    m = kron(m_kernels...)
    P_kernels = getproperty.(x0_kernels, :P)
    P = kron(P_kernels...)

    x0 = Gaussian(m, P)
    As, as, Qs, Hs, hs = broadcast_components((F, q, H), x0, ts, storage)
    emission_projections = (Hs, hs)
    return As, as, Qs, emission_projections, x0
end

_kron_add(A::AbstractMatrix, B::AbstractMatrix) = kron(A, I(size(B,1))) + kron(I(size(A,1)), B)
_kron_add(A::SMatrix{M,M}, B::SMatrix{N,N}) where {M, N} = kron(A, SMatrix{N, N}(I(N))) + kron(SMatrix{M,M}(I(M)), B)

# Sum

function lgssm_components(k::KernelSum, ts::AbstractVector, storage_type::StorageType)
    lgssms = lgssm_components.(k.kernels, Ref(ts), Ref(storage_type))
    As_kernels = getindex.(lgssms, 1)
    as_kernels = getindex.(lgssms, 2)
    Qs_kernels = getindex.(lgssms, 3)
    emission_proj_kernels = getindex.(lgssms, 4)
    x0_kernels = getindex.(lgssms, 5)
    
    As = map(block_diagonal, As_kernels...)
    as = map(vcat, as_kernels...)
    Qs = map(block_diagonal, Qs_kernels...)
    emission_projections = _sum_emission_projections(emission_proj_kernels...)
    x0 = Gaussian(mapreduce(x -> getproperty(x, :m), vcat, x0_kernels), block_diagonal(getproperty.(x0_kernels, :P)...))
    return As, as, Qs, emission_projections, x0
end

function _sum_emission_projections(Hs_hs::Tuple{AbstractVector, AbstractVector}...)
    return map(vcat, first.(Hs_hs)...), sum(last.(Hs_hs))
end

function _sum_emission_projections(
    Cs_cs_Hs_hs::Tuple{AbstractVector, AbstractVector, AbstractVector, AbstractVector}...,
)
    Cs = getindex.(Cs_cs_Hs_hs, 1)
    cs = getindex.(Cs_cs_Hs_hs, 2)
    Hs = getindex.(Cs_cs_Hs_hs, 3)
    hs = getindex.(Cs_cs_Hs_hs, 4)
    C = map(vcat, Cs...)
    c = sum(cs)
    H = map(block_diagonal, Hs...)
    h = map(vcat, hs...)
    return C, c, H, h
end

Base.vcat(x::Zeros{T, 1}, y::Zeros{T, 1}) where {T} = Zeros{T}(length(x) + length(y))

block_diagonal(As::AbstractMatrix{T}...) where {T} = collect(BlockDiagonal(collect(As)))

function block_diagonal(As::SMatrix...)
    M = block_diagonal(map(collect, As)...)
    return SMatrix{sum(map(A -> size(A, 1), As)), sum(map(A -> size(A, 2), As))}(M)
end
